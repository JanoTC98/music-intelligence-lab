-- SQL Server schema for Spotify Music Intelligence persistence
-- SQL Server is an alternative, not the official engine.

USE spotify_app;
GO

CREATE TABLE app_sessions (
    session_id UNIQUEIDENTIFIER NOT NULL,
    started_at DATETIME2(6) NOT NULL CONSTRAINT df_app_sessions_started_at DEFAULT SYSUTCDATETIME(),
    last_seen_at DATETIME2(6) NOT NULL CONSTRAINT df_app_sessions_last_seen DEFAULT SYSUTCDATETIME(),
    app_version VARCHAR(50) NOT NULL,
    environment VARCHAR(20) NOT NULL,
    CONSTRAINT pk_app_sessions PRIMARY KEY (session_id),
    CONSTRAINT chk_app_sessions_environment CHECK (environment IN ('development', 'test', 'production'))
);
GO

CREATE TABLE recommendation_events (
    event_id UNIQUEIDENTIFIER NOT NULL,
    session_id UNIQUEIDENTIFIER NOT NULL,
    created_at DATETIME2(6) NOT NULL CONSTRAINT df_recommendation_events_created_at DEFAULT SYSUTCDATETIME(),
    recommender_type VARCHAR(50) NOT NULL,
    query_recording_group_id CHAR(64) NOT NULL,
    CONSTRAINT pk_recommendation_events PRIMARY KEY (event_id),
    CONSTRAINT fk_recommendation_events_session FOREIGN KEY (session_id) REFERENCES app_sessions(session_id)
);
GO

CREATE TABLE recommendation_results (
    event_id UNIQUEIDENTIFIER NOT NULL,
    result_position SMALLINT NOT NULL,
    recommended_recording_group_id CHAR(64) NOT NULL,
    distance FLOAT NOT NULL,
    similarity FLOAT NOT NULL,
    CONSTRAINT pk_recommendation_results PRIMARY KEY (event_id, result_position),
    CONSTRAINT fk_recommendation_results_event FOREIGN KEY (event_id) REFERENCES recommendation_events(event_id)
);
GO

CREATE TABLE classifier_events (
    event_id UNIQUEIDENTIFIER NOT NULL,
    session_id UNIQUEIDENTIFIER NOT NULL,
    created_at DATETIME2(6) NOT NULL CONSTRAINT df_classifier_events_created_at DEFAULT SYSUTCDATETIME(),
    classifier_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    CONSTRAINT pk_classifier_events PRIMARY KEY (event_id),
    CONSTRAINT fk_classifier_events_session FOREIGN KEY (session_id) REFERENCES app_sessions(session_id)
);
GO

CREATE TABLE classifier_predictions (
    event_id UNIQUEIDENTIFIER NOT NULL,
    prediction_rank SMALLINT NOT NULL,
    genre NVARCHAR(100) NOT NULL,
    score DECIMAL(10, 9) NOT NULL,
    is_top_prediction BIT NOT NULL,
    CONSTRAINT pk_classifier_predictions PRIMARY KEY (event_id, prediction_rank),
    CONSTRAINT fk_classifier_predictions_event FOREIGN KEY (event_id) REFERENCES classifier_events(event_id)
);
GO

CREATE TABLE feedback_events (
    feedback_id UNIQUEIDENTIFIER NOT NULL,
    session_id UNIQUEIDENTIFIER NOT NULL,
    created_at DATETIME2(6) NOT NULL CONSTRAINT df_feedback_events_created_at DEFAULT SYSUTCDATETIME(),
    source_event_id UNIQUEIDENTIFIER NULL,
    source_type VARCHAR(50) NOT NULL,
    rating TINYINT NOT NULL,
    CONSTRAINT pk_feedback_events PRIMARY KEY (feedback_id),
    CONSTRAINT fk_feedback_events_session FOREIGN KEY (session_id) REFERENCES app_sessions(session_id),
    CONSTRAINT fk_feedback_events_source FOREIGN KEY (source_event_id) REFERENCES recommendation_events(event_id),
    CONSTRAINT chk_feedback_events_rating CHECK (rating >= 1 AND rating <= 5)
);
GO