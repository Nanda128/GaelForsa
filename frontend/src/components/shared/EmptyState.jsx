import React from 'react';

function EmptyState({ title, message, className = '' }) {
    return (
        <div className={`empty-state ${className}`}>
            {title && <div className="empty-title">{title}</div>}
            {message && <div className="empty-message">{message}</div>}
        </div>
    );
}

export default EmptyState;

