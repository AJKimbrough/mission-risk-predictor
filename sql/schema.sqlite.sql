WITH base AS (
  SELECT * FROM features WHERE valid_time IS NOT NULL
)
SELECT
  x.id, x.lat, x.lon, x.valid_time,
  x.vis_sm, x.ceiling_ft, x.wind_kts, x.gust_kts, x.xwind_kts, x.pop_pct,
  x.convective_flag, x.icing_flag, x.tfr_active_flag, x.daylight_flag,
  CASE
    WHEN (y.vis_sm IS NOT NULL AND y.vis_sm < 3)
      OR (y.ceiling_ft IS NOT NULL AND y.ceiling_ft < 1000)
      OR (y.wind_kts IS NOT NULL AND y.wind_kts > 25)
      OR (y.gust_kts IS NOT NULL AND y.gust_kts > 35)
      OR (y.tfr_active_flag = 1)
    THEN 1 ELSE 0
  END AS label
FROM base x
LEFT JOIN base y
  ON y.lat = x.lat
 AND y.lon = x.lon
 AND y.valid_time = datetime(x.valid_time, '+3 hours')  
ORDER BY x.valid_time;
