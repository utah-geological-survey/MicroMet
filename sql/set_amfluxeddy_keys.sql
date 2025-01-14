ALTER TABLE amfluxeddy
ADD CONSTRAINT amfluxeddy_stationid_datetime_start_unique
    UNIQUE (stationid, datetime_start);