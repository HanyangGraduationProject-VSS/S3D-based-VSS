import unittest
import tnms
import numpy as np

class TestTemporalNonMaximumSuppression(unittest.TestCase):
    spans = np.array(
        [
            (0, 3),
            (1, 4),
            (2, 4),
            (0.3, 4),
            (0.1, 2.2),
        ]
    )

    scores = np.array(
        [
            0.5, 0.4, 0.2, 0.1, 0.3,
        ]
    )

    def test_IOU(self):
        '''
        Test IoU utility(Intersection over union of temporal spans)
        '''

        self.assertEqual(tnms.IoU((0, 1), (0.5, 2.0)), 0.25)

    def test_non_maximum_suppression(self):
        '''
        Test non-maximum suppression of temporal spans
        '''

        nms_indices = tnms.non_maximum_suppression(self.spans, self.scores)
        self.assertEqual(nms_indices, [0, 2])
    
    def test_get_weighted_span(self):
        '''
        Test get weighted span of temporal spans
        '''

        nms_indices = tnms.non_maximum_suppression(self.spans, self.scores)
        weighted_span = tnms.get_weighted_span(self.spans[nms_indices], self.scores[nms_indices])

        self.assertAlmostEqual(weighted_span[0], 0.85714286, delta=0.00001)
        self.assertAlmostEqual(weighted_span[1], 2.57142857, delta=0.00001)

if __name__ == '__main__':
    unittest.main()