import unittest
import torch.nn as nn
import torch
import torch.nn.functional as F


class TestLosses(unittest.TestCase):
    def test_bfocalloss(self):
        from dlutils.losses.focal import BinaryFocalLoss, \
            BinaryFocalLossWithLogits
        """
        Test some predefines focal loss values
        """

        # examples
        ########################################################################
        # binary values
        p = torch.Tensor([[0, 0.2, 0.5, 1.0], [0, 0.2, 0.5, 1.0]])
        t = torch.Tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        p_l = torch.Tensor([[-2, -1, 0, 2], [-2, -1, 0, 1]])

        ########################################################################
        # params
        gamma = 2
        alpha = 0.25
        eps = 1e-8

        ########################################################################
        # compute targets
        # target for focal loss
        p_t = p * t + (1 - p) * (1 - t)
        alpha_t = torch.Tensor([alpha]).expand_as(t) * t + \
            (1 - t) * (1 - torch.Tensor([alpha]).expand_as(t))
        w = alpha_t * (1 - p_t).pow(torch.Tensor([gamma]))
        fc_value = F.binary_cross_entropy(p, t, w, reduction='none')

        # target for focal loss with logit
        p_tmp = torch.sigmoid(p_l)
        p_t = p_tmp * t + (1 - p_tmp) * (1 - t)
        alpha_t = torch.Tensor([alpha]).expand_as(t) * t + \
            (1 - t) * (1 - torch.Tensor([alpha]).expand_as(t))
        w = alpha_t * (1 - p_t).pow(torch.Tensor([gamma]))

        fc_value_logit = \
            F.binary_cross_entropy_with_logits(p_l, t, w, reduction='none')

        ########################################################################
        # test against BCE and CE =>focal loss with gamma=0, alpha=None
        # test against binary_cross_entropy
        bce = nn.BCELoss(reduction='none')
        focal = BinaryFocalLoss(alpha=None, gamma=0, reduction='none')
        bce_loss = bce(p, t)
        focal_loss = focal(p, t)

        self.assertTrue(torch.allclose(bce_loss, focal_loss, atol=eps))

        # test against binary_cross_entropy with logit
        bce = nn.BCEWithLogitsLoss()
        focal = BinaryFocalLossWithLogits(alpha=None, gamma=0)
        bce_loss = bce(p_l, t)
        focal_loss = focal(p_l, t)
        self.assertTrue(torch.allclose(bce_loss, focal_loss, atol=eps))

        ########################################################################
        # test focal loss with pre computed values
        # test focal loss binary (values manually pre computed)
        focal = BinaryFocalLoss(gamma=gamma, alpha=alpha, reduction='none')
        focal_loss = focal(p, t)
        self.assertTrue(torch.allclose(fc_value, focal_loss, atol=eps))

        # test focal loss binary with logit (values manually pre computed)
        # Note that now p_l is used as prediction
        focal = BinaryFocalLossWithLogits(
            gamma=gamma, alpha=alpha, reduction='none')
        focal_loss = focal(p_l, t)
        self.assertTrue(torch.allclose(focal_loss, fc_value_logit))

        ########################################################################
        # test if backward function works
        p.requires_grad = True
        focal = BinaryFocalLoss(gamma=gamma, alpha=alpha)
        focal_loss = focal(p, t)
        try:
            focal_loss.backward()
        except BaseException:
            self.assertTrue(False, "Backward function failed for focal loss")

        p_l.requires_grad = True
        focal = BinaryFocalLossWithLogits(gamma=gamma, alpha=alpha)
        focal_loss = focal(p_l, t)
        try:
            focal_loss.backward()
        except BaseException:
            self.assertTrue(
                False, "Backward function failed for focal loss with logits")

    def test_focalloss(self):
        """
        Unittest for focal loss
        """
        from dlutils.losses.focal import FocalLoss, FocalLossWithLogits
        weight = torch.Tensor([0.2, 0.2, 0.6])
        loss_fn = FocalLossWithLogits(alpha=weight)
        inp0 = torch.zeros((1, 3, 2, 2), requires_grad=True) + 0.1
        target0 = torch.zeros((1, 2, 2)).to(torch.long)
        loss0 = loss_fn(inp0, target0)
        loss0.backward()
        # TODO add test cases
        # assert (np.isclose(loss0.detach().numpy(), -1.5613277))

    def test_softdice(self):
        """
        Unittest for softdice loss
        """
        from dlutils.losses.soft_dice import SoftDiceLoss

        loss_fn = SoftDiceLoss()
        inp0 = torch.zeros((1, 3, 1, 1), requires_grad=True) + 0.1
        target0 = torch.zeros((1, 1, 1), dtype=torch.long) + 1
        loss0 = loss_fn(inp0, target0)
        loss0.backward()
        self.assertTrue(torch.isclose(loss0, torch.tensor(-0.79653674364089)))


if __name__ == '__main__':
    unittest.main()
