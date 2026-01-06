import React from 'react';

function ErrorState({ message, onRetry, className = '' }) {
    return (
        <div className={`error-state ${className}`}>
            <div className="error-icon">!</div>
            <div className="error-message">{message}</div>
            {onRetry && (
                <button className="btn btn-primary" onClick={onRetry} type="button">
                    Retry
                </button>
            )}
        </div>
    );
}

export default ErrorState;

