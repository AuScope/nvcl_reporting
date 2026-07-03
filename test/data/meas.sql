CREATE TABLE public.meas (
    report_category text NOT NULL,
    provider text NOT NULL,
    nvcl_id text NOT NULL,
    log_id text NOT NULL,
    algorithm text NOT NULL,
    log_type text NOT NULL,
    algorithm_id text NOT NULL,
    borehole_id text NOT NULL,
    drill_hole_name text NOT NULL,
    easting double precision NOT NULL,
    northing double precision NOT NULL,
    crs text NOT NULL,
    start_depth double precision NOT NULL,
    end_depth double precision NOT NULL,
    has_vnir boolean NOT NULL,
    has_swir boolean NOT NULL,
    has_tir boolean NOT NULL,
    has_mir boolean NOT NULL,
    modified_datetime date,
    minerals text NOT NULL,
    mincnts text NOT NULL,
    data text NOT NULL
);


ALTER TABLE ONLY public.meas
    ADD CONSTRAINT meas_pkey PRIMARY KEY (report_category, provider, nvcl_id, log_id, algorithm, log_type, algorithm_id);

