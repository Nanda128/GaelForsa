import React from 'react';

function PageHeader({ title, subtitle, actions, className = '' }) {
    return (
        <div className={`dashboard-page-header ${className}`}>
            <div>
                <h1>{title}</h1>
                {subtitle && <div className="dashboard-subtitle">{subtitle}</div>}
            </div>
            {actions && <div className="dashboard-actions">{actions}</div>}
        </div>
    );
}

export default PageHeader;

