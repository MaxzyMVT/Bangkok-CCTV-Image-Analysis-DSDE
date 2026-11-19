import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DataEngineering" / "labeling"))

from fix_overlaps import fix_frame  # noqa: E402
from review_store import ReviewStore  # noqa: E402

STEM = "cam0001_img00001"
CAR, TRUCK = 0, 3


class FixFrameTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ReviewStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _reviewed(self, pre, labels):
        self.store.write_prelabel("train", STEM, pre)
        self.store.save("train", STEM, labels, "save")

    def test_clear_double_box_is_merged_and_saved(self):
        self._reviewed([(CAR, .5, .5, .2, .2, .8), (TRUCK, .505, .5, .21, .2, .5)],
                       [(CAR, .5, .5, .2, .2), (TRUCK, .505, .5, .21, .2)])
        self.assertEqual(fix_frame(self.store, "train", STEM, apply=True), 1)
        self.assertEqual([b[0] for b in self.store.read_labels("train", STEM)], [CAR])

    def test_dry_run_writes_nothing(self):
        self._reviewed([(CAR, .5, .5, .2, .2, .8), (TRUCK, .505, .5, .21, .2, .5)],
                       [(CAR, .5, .5, .2, .2), (TRUCK, .505, .5, .21, .2)])
        self.assertEqual(fix_frame(self.store, "train", STEM, apply=False), 1)
        self.assertEqual(len(self.store.read_labels("train", STEM)), 2)

    def test_doubtful_pair_is_left_for_a_person(self):
        self._reviewed([(CAR, .5, .5, .2, .2, .4), (TRUCK, .505, .5, .21, .2, .7)],
                       [(CAR, .5, .5, .2, .2), (TRUCK, .505, .5, .21, .2)])
        self.assertEqual(fix_frame(self.store, "train", STEM, apply=True), 0)
        self.assertEqual(len(self.store.read_labels("train", STEM)), 2)

    def test_box_a_person_reclassed_is_not_overruled(self):
        # the reviewer changed the truck box to bus; it no longer matches a pre-label box, so no confidence:
        # car + bus with an unknown score must stay for a person
        self._reviewed([(CAR, .5, .5, .2, .2, .8), (TRUCK, .505, .5, .21, .2, .5)],
                       [(CAR, .5, .5, .2, .2), (2, .505, .5, .21, .2)])
        self.assertEqual(fix_frame(self.store, "train", STEM, apply=True), 0)

    def test_reclassed_twin_of_same_class_is_merged(self):
        # truck re-classed to car but the old car box left in place: car + car on one vehicle
        self._reviewed([(CAR, .5, .5, .2, .2, .4), (TRUCK, .505, .5, .21, .2, .7)],
                       [(CAR, .5, .5, .2, .2), (CAR, .505, .5, .21, .2)])
        self.assertEqual(fix_frame(self.store, "train", STEM, apply=True), 1)
        self.assertEqual(len(self.store.read_labels("train", STEM)), 1)


if __name__ == "__main__":
    unittest.main()
