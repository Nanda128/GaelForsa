import React from 'react';

function StatCard({ label, value, unit, variant = 'default', className = '' }) {
    const variantClasses = {
        default: '',
        success: 'success',
        warning: 'warning',
        danger: 'danger',
        info: 'info'
    };
    // const variantClass = variantClasses[variant] || '';

    return (
        <div className={`stat-card ${variantClasses[variant]} ${className}`}>
            <div className="stat-label">{label}</div>
            <div className={`stat-value-large ${variantClasses[variant]}`}>{value}</div>
            {unit && <div className="stat-unit">{unit}</div>}
        </div>
    );
}

export default StatCard;

