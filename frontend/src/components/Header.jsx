import React, { useMemo } from 'react';
import { useTheme } from '../contexts/ThemeContext';

function Header({ turbines = [], onRefresh, onDashboardClick, onMapClick, currentView = 'map', onToggleSidebar, isSidebarOpen = true }) {
    const { theme, toggleTheme } = useTheme();
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
        <header className={`header ${isSidebarOpen ? '' : 'collapsed'}`} role="banner">
            <div className="header-top">
                <button
                    className="header-btn-theme"
                    onClick={toggleTheme}
                    aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                    type="button"
                    title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                    {theme === 'dark' ? (
                        <svg className="header-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                            <circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                        </svg>
                    ) : (
                        <svg className="header-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                        </svg>
                    )}
                </button>
                <button
                    className="header-toggle"
                    onClick={() => onToggleSidebar?.()}
                    aria-label={isSidebarOpen ? 'Collapse sidebar' : 'Open sidebar'}
                    aria-pressed={!!isSidebarOpen}
                    type="button"
                >
                    <span className="toggle-icon" aria-hidden="true">{isSidebarOpen ? '‹' : '›'}</span>
                </button>
            </div>
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
                    className={`header-btn ${currentView === 'dashboard' ? 'active' : ''}`}
                    onClick={onDashboardClick} 
                    aria-label="View dashboard"
                    type="button"
                >
                    <span className="header-btn-label">Dashboard</span>
                    <span className="header-btn-icon" aria-hidden>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
                    </span>
                </button>
                <button 
                    className={`header-btn ${currentView === 'map' ? 'active' : ''}`}
                    onClick={onMapClick} 
                    aria-label="View map"
                    type="button"
                >
                    <span className="header-btn-label">Map</span>
                    <span className="header-btn-icon" aria-hidden>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>
                    </span>
                </button>
                <button 
                    className="header-btn" 
                    onClick={onRefresh} 
                    aria-label="Refresh data"
                    type="button"
                >
                    <span className="header-btn-label">Refresh</span>
                    <span className="header-btn-icon" aria-hidden>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
                    </span>
                </button>
            </div>
            <div className="header-brand">
                <a href="#" className="header-logo" aria-label="GaelFórsa Home" onClick={(e) => { e.preventDefault(); onMapClick(); }}>
                    <div className="header-logo-icon">GF</div>
                    <span>GaelFórsa</span>
                </a>
            </div>
        </header>
    );
}

export default Header;
