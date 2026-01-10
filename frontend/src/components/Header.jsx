import React, { useMemo } from 'react';

function Header({ turbines = [], onRefresh, onDashboardClick, onMapClick, currentView = 'map' }) {
    const stats = useMemo(() => {
        const total = turbines.length;
        // const statusCounts = { green: 0, yellow: 0, red: 0 };
        // turbines.forEach(t => statusCounts[t.status]++);
        const healthy = turbines.filter(t => t.status === 'green').length;
        const warning = turbines.filter(t => t.status === 'yellow').length;
        const critical = turbines.filter(t => t.status === 'red').length;
        // console.log('Header stats:', { total, healthy, warning, critical });
        return { total, healthy, warning, critical };
    }, [turbines]);

    return (
        <header className="header">
            <a href="#" className="header-logo" aria-label="GaelFórsa Home" onClick={(e) => { e.preventDefault(); onMapClick(); }}>
                <div className="header-logo-icon">GF</div>
                <span>GaelFórsa</span>
            </a>
            <nav className="header-nav">
                <div className="header-stats">
                    <div className="stat-item">
                        <span className="stat-label">Total</span>
                        <span className="stat-value">{stats.total}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Healthy</span>
                        <span className="stat-value success">{stats.healthy}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Warning</span>
                        <span className="stat-value warning">{stats.warning}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Critical</span>
                        <span className="stat-value danger">{stats.critical}</span>
                    </div>
                </div>
            </nav>
            <div className="header-actions">
                <button 
                    className={`header-btn ${currentView === 'map' ? 'active' : ''}`}
                    onClick={onMapClick} 
                    aria-label="View map"
                    type="button"
                >
                    <span>Map</span>
                </button>
                <button 
                    className={`header-btn primary ${currentView === 'dashboard' ? 'active' : ''}`}
                    onClick={onDashboardClick} 
                    aria-label="View dashboard"
                    type="button"
                >
                    <span>Dashboard</span>
                </button>
                <button 
                    className="header-btn" 
                    onClick={onRefresh} 
                    aria-label="Refresh data"
                    type="button"
                >
                    <span>Refresh</span>
                </button>
            </div>
        </header>
    );
}

export default Header;
