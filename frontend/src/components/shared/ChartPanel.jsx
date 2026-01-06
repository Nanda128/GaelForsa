import React from 'react';

function ChartPanel({ title, subtitle, children, className = '' }) {
    return (
        <div className={`chart-panel ${className}`}>
            <div className="chart-header">
                <h3>{title}</h3>
                {subtitle && <div className="chart-subtitle">{subtitle}</div>}
            </div>
            {children}
        </div>
    );
}

export default ChartPanel;

