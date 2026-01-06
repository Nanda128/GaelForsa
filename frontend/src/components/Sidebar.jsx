import React, { useState, useMemo } from 'react';
import { getStatusLabel } from '../utils/helpers';

function Sidebar({ turbines = [], statusFilters, onFilterChange, onTurbineClick }) {
    const [collapsed, setCollapsed] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');

    const filteredTurbines = useMemo(() => {
        // const filtered = [];
        // turbines.forEach(t => { if (statusFilters[t.status] && (!searchQuery || t.name.toLowerCase().includes(searchQuery.toLowerCase()))) filtered.push(t); });
        return turbines.filter(turbine => {
            if (!statusFilters[turbine.status]) {
                return false;
            }
            if (searchQuery) {
                const query = searchQuery.toLowerCase();
                const name = turbine.name.toLowerCase();
                // return name.startsWith(query); // tried startsWith but includes is better
                return name.includes(query);
            }
            return true;
        });
    }, [turbines, statusFilters, searchQuery]);

    const stats = useMemo(() => {
        const total = turbines.length;
        const green = turbines.filter(t => t.status === 'green').length;
        const yellow = turbines.filter(t => t.status === 'yellow').length;
        const red = turbines.filter(t => t.status === 'red').length;
        const healthPercentage = total > 0 ? ((green / total) * 100).toFixed(0) : 0;
        return { total, green, yellow, red, healthPercentage };
    }, [turbines]);

    const handleFilterToggle = (status) => {
        onFilterChange({
            ...statusFilters,
            [status]: !statusFilters[status]
        });
    };

    return (
        <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
            <button
                className="sidebar-toggle"
                onClick={() => setCollapsed(!collapsed)}
                aria-label="Toggle sidebar"
                type="button"
            >
                <span className="toggle-icon">☰</span>
            </button>
            <div className="sidebar-content">
                <div className="sidebar-header">
                    <h2 className="sidebar-title">Turbines</h2>
                    <div className="sidebar-count">{stats.total}</div>
                </div>
                <div className="search-container">
                    <label htmlFor="turbine-search" className="visually-hidden">
                        Search turbines by name
                    </label>
                    <input
                        type="text"
                        id="turbine-search"
                        className="search-input"
                        placeholder="Search turbines..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        aria-label="Search turbines by name"
                    />
                </div>
                <div className="sidebar-quick-stats">
                    <div className="quick-stat-card success">
                        <div className="quick-stat-value">{stats.healthPercentage}%</div>
                        <div className="quick-stat-label">Health Rate</div>
                    </div>
                    <div className={`quick-stat-card ${stats.red > 0 ? 'danger' : stats.yellow > 0 ? 'warning' : 'success'}`}>
                        <div className="quick-stat-value">{stats.red + stats.yellow}</div>
                        <div className="quick-stat-label">Needs Attention</div>
                    </div>
                </div>
                <div className="filter-container">
                    <h3 className="filter-title">Status Filter</h3>
                    <div className="filter-options">
                        <label className="filter-option">
                            <input
                                type="checkbox"
                                checked={statusFilters.green}
                                onChange={() => handleFilterToggle('green')}
                            />
                            <span className="status-indicator green"></span>
                            <span>Healthy</span>
                            <span className="filter-count">{stats.green}</span>
                        </label>
                        <label className="filter-option">
                            <input
                                type="checkbox"
                                checked={statusFilters.yellow}
                                onChange={() => handleFilterToggle('yellow')}
                            />
                            <span className="status-indicator yellow"></span>
                            <span>Warning</span>
                            <span className="filter-count">{stats.yellow}</span>
                        </label>
                        <label className="filter-option">
                            <input
                                type="checkbox"
                                checked={statusFilters.red}
                                onChange={() => handleFilterToggle('red')}
                            />
                            <span className="status-indicator red"></span>
                            <span>Critical</span>
                            <span className="filter-count">{stats.red}</span>
                        </label>
                    </div>
                </div>
                <div className="turbine-list">
                        {filteredTurbines.length === 0 ? (
                            <div className="empty-state-sidebar">
                                <div className="empty-text">No turbines match your filters</div>
                            </div>
                        ) : (
                        filteredTurbines.map(turbine => (
                            <div
                                key={turbine.id}
                                className="turbine-item"
                                onClick={() => onTurbineClick(turbine)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' || e.key === ' ') {
                                        e.preventDefault();
                                        onTurbineClick(turbine);
                                    }
                                }}
                                tabIndex={0}
                                role="button"
                                aria-label={`View details for ${turbine.name}`}
                            >
                                <span className={`status-dot ${turbine.status}`}></span>
                                <div className="turbine-item-content">
                                    <div className="turbine-item-name">{turbine.name}</div>
                                    <div className="turbine-item-status">
                                        {getStatusLabel(turbine.status)}
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </aside>
    );
}

export default Sidebar;
