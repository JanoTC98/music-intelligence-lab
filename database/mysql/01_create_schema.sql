-- MySQL schema for Spotify Music Intelligence persistence
-- MySQL >= 8.0.16 required (CHECK constraints)

CREATE TABLE IF NOT EXISTS app_sessions (
    session_id CHAR(36) NOT NULL,
    started_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    last_seen_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    app_version VARCHAR(50) NOT NULL,
    environment VARCHAR(20) NOT NULL,
    PRIMARY KEY (session_id),
    INDEX idx_app_sessions_last_seen (last_seen_at),
    CONSTRAINT chk_app_sessions_environment
        CHECK (environment IN ('development', 'test', 'production'))
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS recommendation_events (
    event_id CHAR(36) NOT NULL,
    session_id CHAR(36) NOT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    recommendation_type VARCHAR(20) NOT NULL,
    selected_track_id VARCHAR(100) NULL,
    selected_recording_group_id CHAR(64) NULL,
    selected_preset VARCHAR(100) NULL,
    query_payload JSON NOT NULL,
    requested_result_count SMALLINT UNSIGNED NOT NULL,
    returned_result_count SMALLINT UNSIGNED NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    latency_ms INT UNSIGNED NOT NULL,
    PRIMARY KEY (event_id),
    INDEX idx_recommendation_events_created_at (created_at),
    INDEX idx_recommendation_events_session (session_id),
    INDEX idx_recommendation_events_model (model_version),
    CONSTRAINT fk_recommendation_events_session
        FOREIGN KEY (session_id)
        REFERENCES app_sessions(session_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_recommendation_type
        CHECK (recommendation_type IN ('track', 'preferences')),
    CONSTRAINT chk_requested_result_count
        CHECK (requested_result_count BETWEEN 1 AND 100),
    CONSTRAINT chk_returned_result_count
        CHECK (returned_result_count BETWEEN 0 AND 100)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS recommendation_results (
    event_id CHAR(36) NOT NULL,
    result_position SMALLINT UNSIGNED NOT NULL,
    recommended_recording_group_id CHAR(64) NOT NULL,
    recommended_track_id VARCHAR(100) NOT NULL,
    similarity_score DECIMAL(10, 9) NULL,
    distance_score DECIMAL(14, 9) NULL,
    explanation_payload JSON NULL,
    PRIMARY KEY (event_id, result_position),
    INDEX idx_recommendation_results_recording (recommended_recording_group_id),
    CONSTRAINT fk_recommendation_results_event
        FOREIGN KEY (event_id)
        REFERENCES recommendation_events(event_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_result_position
        CHECK (result_position BETWEEN 1 AND 100),
    CONSTRAINT chk_similarity_score
        CHECK (similarity_score IS NULL OR similarity_score BETWEEN -1.0 AND 1.0),
    CONSTRAINT chk_distance_score
        CHECK (distance_score IS NULL OR distance_score >= 0.0)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS classifier_events (
    event_id CHAR(36) NOT NULL,
    session_id CHAR(36) NOT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    classifier_type VARCHAR(20) NOT NULL,
    selected_track_id VARCHAR(100) NOT NULL,
    selected_recording_group_id CHAR(64) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    true_labels_payload JSON NULL,
    threshold_value DECIMAL(8, 6) NULL,
    latency_ms INT UNSIGNED NOT NULL,
    PRIMARY KEY (event_id),
    INDEX idx_classifier_events_created_at (created_at),
    INDEX idx_classifier_events_session (session_id),
    INDEX idx_classifier_events_model (model_version),
    CONSTRAINT fk_classifier_events_session
        FOREIGN KEY (session_id)
        REFERENCES app_sessions(session_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_classifier_type
        CHECK (classifier_type IN ('multilabel', 'multiclass')),
    CONSTRAINT chk_threshold_value
        CHECK (threshold_value IS NULL OR threshold_value BETWEEN 0.0 AND 1.0)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS classifier_predictions (
    event_id CHAR(36) NOT NULL,
    prediction_rank SMALLINT UNSIGNED NOT NULL,
    genre VARCHAR(100) NOT NULL,
    score DECIMAL(10, 9) NOT NULL,
    passed_threshold BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (event_id, prediction_rank),
    INDEX idx_classifier_predictions_genre (genre),
    CONSTRAINT fk_classifier_predictions_event
        FOREIGN KEY (event_id)
        REFERENCES classifier_events(event_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_prediction_rank
        CHECK (prediction_rank BETWEEN 1 AND 114),
    CONSTRAINT chk_prediction_score
        CHECK (score BETWEEN 0.0 AND 1.0)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id CHAR(36) NOT NULL,
    session_id CHAR(36) NOT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    source_event_id CHAR(36) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    target_recording_group_id CHAR(64) NULL,
    feedback_value TINYINT NOT NULL,
    reason_code VARCHAR(50) NULL,
    PRIMARY KEY (feedback_id),
    INDEX idx_feedback_events_created_at (created_at),
    INDEX idx_feedback_events_source (source_event_id, source_type),
    CONSTRAINT fk_feedback_events_session
        FOREIGN KEY (session_id)
        REFERENCES app_sessions(session_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_feedback_source_type
        CHECK (source_type IN ('recommendation_list', 'recommendation_item', 'classifier')),
    CONSTRAINT chk_feedback_value
        CHECK (feedback_value IN (-1, 1))
) ENGINE=InnoDB;
