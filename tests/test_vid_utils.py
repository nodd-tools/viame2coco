import unittest
import cv2
from viame2coco import vid_utils


class TestVideoUtils(unittest.TestCase):
    """Unit tests for viame2coco.vid_utils functions."""

    def setUp(self):
        """Set up the video capture object for testing."""
        self.video_path = "tests/example.mp4"
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open test video: {self.video_path}")


    def tearDown(self):
        """Release the video capture object."""
        self.cap.release()

    def test_find_last_valid_timestamp(self):
        
        MS_PER_S = 1000
        lower_ms = 14.4 * MS_PER_S
        upper_ms = 14.6 * MS_PER_S

        last_ts = vid_utils.find_last_valid_timestamp(self.cap, lower_ms, upper_ms)

        #print(f"Detected last valid timestamp: {last_ts:.3f} ms")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        duration_milliseconds = duration * 1000 # 1k milliseconds in a second

        #print(f"'Last frame' computed timestamp: {duration_milliseconds:.3f} ms")

        self.cap.set(cv2.CAP_PROP_POS_MSEC, last_ts) # the end of the film
        last_frame_success, _ = self.cap.read()

        too_far = 14466.667 # ms
        self.cap.set(cv2.CAP_PROP_POS_MSEC, too_far) # the end of the film
        too_far_success, _ = self.cap.read()
        
        #print(f"image read success: {last_frame_success}")

        self.assertAlmostEqual(last_ts, 14433.333, 2)
        assert(last_frame_success)
        assert(not too_far_success)


if __name__ == "__main__":
    unittest.main()

