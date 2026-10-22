import unittest

import pandas as pd

from update_pipeline import attach_view_history


class ViewHistoryTests(unittest.TestCase):
    def test_repeat_snapshot_calculates_delta_and_hourly_rate(self):
        old = pd.DataFrame([{
            "id": "channel_10",
            "views": "100",
            "scraped_at": "2026-10-22T10:00:00+00:00",
        }])
        new = pd.DataFrame([{
            "id": "channel_10",
            "views": "160",
            "scraped_at": "2026-10-22T11:00:00+00:00",
        }])
        row = attach_view_history(old, new).iloc[0]
        self.assertEqual(row["previous_views"], 100)
        self.assertEqual(row["views_delta"], 60)
        self.assertEqual(row["views_velocity_per_hour"], 60.0)
        self.assertEqual(row["views_interval_hours"], 1.0)

    def test_first_snapshot_does_not_invent_velocity(self):
        new = pd.DataFrame([{
            "id": "channel_11",
            "views": "2",
            "scraped_at": "2026-10-22T11:00:00+00:00",
        }])
        row = attach_view_history(pd.DataFrame(), new).iloc[0]
        self.assertTrue(pd.isna(row["views_delta"]))
        self.assertTrue(pd.isna(row["views_velocity_per_hour"]))

    def test_negative_rounded_change_is_not_scored(self):
        old = pd.DataFrame([{
            "id": "channel_12",
            "views": "1000",
            "scraped_at": "2026-10-22T10:00:00+00:00",
        }])
        new = pd.DataFrame([{
            "id": "channel_12",
            "views": "999",
            "scraped_at": "2026-10-22T11:00:00+00:00",
        }])
        row = attach_view_history(old, new).iloc[0]
        self.assertTrue(pd.isna(row["views_delta"]))
        self.assertTrue(pd.isna(row["views_velocity_per_hour"]))

    def test_stale_snapshot_is_not_used_for_hourly_rate(self):
        old = pd.DataFrame([{
            "id": "channel_13",
            "views": "100",
            "scraped_at": "2026-10-18T10:00:00+00:00",
        }])
        new = pd.DataFrame([{
            "id": "channel_13",
            "views": "900",
            "scraped_at": "2026-10-22T10:00:00+00:00",
        }])
        row = attach_view_history(old, new).iloc[0]
        self.assertTrue(pd.isna(row["views_delta"]))
        self.assertTrue(pd.isna(row["views_velocity_per_hour"]))


if __name__ == "__main__":
    unittest.main()
