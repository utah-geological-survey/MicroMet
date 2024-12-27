CREATE OR REPLACE VIEW most_recent_stations_met AS
SELECT DISTINCT ON (stationid)
       stationid,
       datetime_start
FROM amfluxmet
ORDER BY stationid, datetime_start DESC;