"""Triplet loss and the identity check behind its mining, on hand-built embeddings."""

import importlib.util
import unittest

# The fixture-only CI job installs Pillow alone; the dependency jobs run these tests.
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this environment")

import torch

from facial_attendance.recognition.metric import check_triplets, triplet_loss


def pk_targets(p=4, k=3):
    return torch.arange(p).repeat_interleave(k)


class TripletLossTests(unittest.TestCase):
    def test_hand_computed_loss_on_four_points(self):
        # Identity 0 at angles 0 and 90 degrees, identity 1 at 180 and 270 degrees.
        e = torch.tensor([[1.0, 0], [0, 1.0], [-1.0, 0], [0, -1.0]])
        t = torch.tensor([0, 0, 1, 1])
        loss, stats = triplet_loss(e, t, margin=0.2)
        # Every d(a,p) = 2; each anchor has one negative at 2 and one at 4.
        # Semi-hard picks the negative at 4 (closest beyond 2): max(0, 2 - 4 + 0.2) = 0.
        self.assertEqual(stats["anchor_positive_pairs"], 4)
        self.assertEqual(stats["fallback"], 0)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_collapse_gives_margin_and_zero_spread(self):
        loss, stats = triplet_loss(torch.ones(12, 8), pk_targets(), margin=0.2)
        self.assertAlmostEqual(loss.item(), 0.2, places=6)
        self.assertAlmostEqual(stats["embedding_std"], 0.0, places=6)
        self.assertEqual(stats["fallback"], stats["anchor_positive_pairs"])

    def test_well_separated_identities_give_zero_loss(self):
        t = pk_targets(p=4, k=3)
        e = torch.nn.functional.one_hot(t, 8).float() + 0.01 * torch.randn(12, 8)
        loss, stats = triplet_loss(e, t)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertEqual(stats["active"], 0)

    def test_semi_hard_negative_is_counted(self):
        # Positive at 90 degrees; a negative between it and the margin beyond it.
        e = torch.tensor([[1.0, 0], [0, 1.0], [-0.1, 0.995], [-1.0, 0]])
        t = torch.tensor([0, 0, 1, 1])
        loss, stats = triplet_loss(e, t, margin=0.5)
        self.assertGreater(stats["semi_hard"], 0)
        self.assertGreater(loss.item(), 0)

    def test_mined_relationships_on_random_pk_batches(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                t = pk_targets(p=8, k=4)[torch.randperm(32)]
                e = torch.randn(32, 16, requires_grad=True)
                loss, stats = triplet_loss(e, t)
                # 8 identities x 4 images x 3 positives each.
                self.assertEqual(stats["anchor_positive_pairs"], 8 * 4 * 3)
                loss.backward()
                self.assertTrue(torch.isfinite(e.grad).all())

    def test_batch_without_negatives_or_positives_is_rejected(self):
        for targets in (torch.zeros(4, dtype=torch.long), torch.tensor([0, 0, 1, 2])):
            with self.subTest(targets=targets.tolist()), self.assertRaises(ValueError):
                triplet_loss(torch.randn(len(targets), 4), targets)

    def test_check_triplets_rejects_wrong_relationships(self):
        t = torch.tensor([0, 0, 1, 1])
        a, p = torch.tensor([0]), torch.tensor([1])
        check_triplets(t, a, p, torch.tensor([2]))
        with self.assertRaises(AssertionError):
            check_triplets(t, a, p, torch.tensor([1]))  # negative shares the identity
        with self.assertRaises(AssertionError):
            check_triplets(t, a, torch.tensor([2]), torch.tensor([3]))  # wrong positive
        with self.assertRaises(AssertionError):
            check_triplets(t, a, a, torch.tensor([2]))  # anchor as its own positive


if __name__ == "__main__":
    unittest.main()
