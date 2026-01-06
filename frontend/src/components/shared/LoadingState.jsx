import React from 'react';

function LoadingState({ message = 'Loading...', className = '' }) {
    return (
        <div className={`loading-state ${className}`}>
            <div className="loading-spinner"></div>
            <div className="loading-message">{message}</div>
        </div>
    );
}

export default LoadingState;

