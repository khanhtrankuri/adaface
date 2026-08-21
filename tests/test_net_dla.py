import unittest

import torch
from torch import nn

import net


class DLABackboneTest(unittest.TestCase):
    def test_dla_blocks_never_place_batch_norm_before_weighted_layer(self):
        model = net.build_model('ir_18_dla')

        for module in model.modules():
            if not isinstance(module, nn.Sequential):
                continue
            children = list(module.children())
            for current, following in zip(children, children[1:]):
                is_batch_norm = isinstance(
                    current, (nn.BatchNorm1d, nn.BatchNorm2d))
                is_weighted = isinstance(following, (nn.Conv2d, nn.Linear))
                self.assertFalse(
                    is_batch_norm and is_weighted,
                    '{} contains {} -> {}'.format(
                        module, type(current).__name__, type(following).__name__))

    def test_legacy_layout_is_unchanged(self):
        model = net.build_model('ir_18')
        first_block = model.body[0]

        self.assertIsInstance(first_block, net.BasicBlockIR)
        self.assertIsInstance(first_block.res_layer[0], nn.BatchNorm2d)
        self.assertIsInstance(first_block.res_layer[1], nn.Conv2d)

    def test_dla_forward_and_export_wrapper(self):
        model = net.build_model('ir_18_dla').eval()
        export_model = net.DLAEmbeddingModel(model).eval()
        inputs = torch.randn(2, 3, 112, 112)

        with torch.no_grad():
            raw_embedding = export_model(inputs)
            normalized_embedding, norms = model(inputs)

        self.assertEqual(raw_embedding.shape, (2, 512))
        self.assertEqual(normalized_embedding.shape, (2, 512))
        self.assertEqual(norms.shape, (2, 1))
        torch.testing.assert_close(
            torch.norm(normalized_embedding, p=2, dim=1),
            torch.ones(2),
            rtol=1e-5,
            atol=1e-6)

    def test_export_wrapper_rejects_legacy_backbone(self):
        with self.assertRaisesRegex(ValueError, r'\*_dla backbone'):
            net.DLAEmbeddingModel(net.build_model('ir_18'))

    def test_legacy_checkpoint_can_initialize_dla_model(self):
        legacy_model = net.build_model('ir_18')
        dla_model = net.build_model('ir_18_dla')
        converted = net.convert_legacy_state_dict_for_dla(
            legacy_model.state_dict())

        dla_model.load_state_dict(converted, strict=True)
        torch.testing.assert_close(
            dla_model.body[0].res_layer[0].weight,
            legacy_model.body[0].res_layer[1].weight)


if __name__ == '__main__':
    unittest.main()
