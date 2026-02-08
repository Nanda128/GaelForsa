import React, { useMemo } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useDashboardData } from '../hooks/useDashboardData';
import { exportFarmStatsToCSV, exportPowerDataToCSV } from '../utils/export';
import StatCard from './shared/StatCard';
import ChartPanel from './shared/ChartPanel';
import PageHeader from './shared/PageHeader';
import Button from './shared/Button';
import LoadingState from './shared/LoadingState';
import ErrorState from './shared/ErrorState';

function Dashboard({ turbines = [], onGridImpactClick, onEconomicsClick }) {
    const { farmStats, powerData, healthTrends, loading, error } = useDashboardData(turbines);

    const stats = useMemo(() => {
        if (farmStats) {
            return {
                totalTurbines: farmStats.turbines.total,
                healthyCount: farmStats.turbines.by_status.green,
                warningCount: farmStats.turbines.by_status.yellow,
                criticalCount: farmStats.turbines.by_status.red,
                healthPercentage: parseFloat(farmStats.turbines.health_percentage.toFixed(0)),
                averageHealthScore: farmStats.health_metrics.average_health_score,
                activeAlerts: farmStats.alerts.active,
                criticalAlerts: farmStats.alerts.critical
            };
        }
        
        // Fallback calculation
        const totalTurbines = turbines.length;
        const healthyCount = turbines.filter(t => t.status === 'green').length;
        const warningCount = turbines.filter(t => t.status === 'yellow').length;
        const criticalCount = turbines.filter(t => t.status === 'red').length;
        // const statusCounts = turbines.reduce((acc, t) => { acc[t.status] = (acc[t.status] || 0) + 1; return acc; }, {});
        const healthPercentage = totalTurbines > 0 
            ? ((healthyCount / totalTurbines) * 100).toFixed(0)
            : 0;
        
        return {
            totalTurbines,
            healthyCount,
            warningCount,
            criticalCount,
            healthPercentage: parseFloat(healthPercentage),
            averageHealthScore: null,
            activeAlerts: 0,
            criticalAlerts: 0
        };
    }, [turbines, farmStats]);

    const statusData = [
        { name: 'Healthy', value: stats.healthyCount, color: '#10b981' },
        { name: 'Warning', value: stats.warningCount, color: '#f59e0b' },
        { name: 'Critical', value: stats.criticalCount, color: '#ef4444' }
    ];

    const powerChartData = powerData.map(item => {
        const name = item.turbineName || item.turbine_name || 'Unknown';
        // const truncated = name.length > 15 ? name.substring(0, 15) + '...' : name;
        return {
            name: name.length > 15 ? name.substring(0, 15) + '...' : name,
            average: item.average || item.average_kw || 0,
            max: item.max || item.max_kw || 0
        };
    });

    if (loading) {
        return (
            <main className="dashboard-page">
                <LoadingState message="Loading dashboard statistics..." />
            </main>
        );
    }

    if (error) {
        return (
            <main className="dashboard-page">
                <ErrorState message={error} />
            </main>
        );
    }

    const headerActions = (
        <>
            {onGridImpactClick && (
                <Button variant="secondary" className="dashboard-action-feature" onClick={onGridImpactClick}>
                    Grid Impact Analysis
                </Button>
            )}
            {onEconomicsClick && (
                <Button variant="secondary" className="dashboard-action-feature" onClick={onEconomicsClick}>
                    Economic Analysis
                </Button>
            )}
            {farmStats && (
                <Button 
                    variant="secondary" 
                    onClick={() => exportFarmStatsToCSV(farmStats)}
                    title="Export statistics to CSV"
                >
                    Export Stats
                </Button>
            )}
            {powerChartData.length > 0 && (
                <Button 
                    variant="secondary" 
                    onClick={() => exportPowerDataToCSV(powerData)}
                    title="Export power data to CSV"
                >
                    Export Power Data
                </Button>
            )}
        </>
    );

    return (
        <main className="dashboard-page">
            <PageHeader
                title="Wind Farm Monitoring Dashboard"
                subtitle="Real-time system status and analytics"
                actions={headerActions}
            />
            <div className="dashboard-page-content">
                <div className="dashboard-grid">
                    <div className="dashboard-cell dashboard-cell-rect dashboard-cell-stat-hero">
                        <StatCard
                            label="Total Turbines"
                            value={stats.totalTurbines}
                            unit="units"
                        />
                    </div>
                    <div className="dashboard-cell dashboard-cell-rect">
                        <StatCard
                            label="Operational"
                            value={stats.healthyCount}
                            unit={`${stats.healthPercentage}% availability`}
                            variant="success"
                        />
                    </div>
                    <div className="dashboard-cell dashboard-cell-rect">
                        <StatCard
                            label="Needs Attention"
                            value={stats.warningCount + stats.criticalCount}
                            unit="warning + critical"
                            variant={stats.criticalCount > 0 ? 'danger' : 'warning'}
                        />
                    </div>

                    <div className="dashboard-cell dashboard-cell-square dashboard-cell-square-top-left dashboard-cell-square-span">
                        {healthTrends.length > 0 ? (
                            <ChartPanel 
                                title="Health Trends"
                                subtitle="30-day health score and failure risk"
                                className="dashboard-chart-panel"
                            >
                                <ResponsiveContainer width="100%" height="100%" minHeight={160}>
                                    <LineChart data={healthTrends}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dashboard-chart-grid" />
                                        <XAxis 
                                            dataKey="date" 
                                            angle={-45} 
                                            textAnchor="end" 
                                            height={72}
                                            stroke="#64748b"
                                            tick={{ fontSize: 11 }}
                                            className="dashboard-chart-axis"
                                        />
                                        <YAxis 
                                            stroke="#64748b"
                                            tick={{ fontSize: 11 }}
                                            label={{ value: '(%)', angle: -90, position: 'insideLeft', style: { fill: '#64748b', fontSize: 11 } }}
                                            className="dashboard-chart-axis"
                                        />
                                        <Tooltip 
                                            contentStyle={{ 
                                                backgroundColor: '#ffffff', 
                                                border: '1px solid #cbd5e1',
                                                borderRadius: '8px',
                                                fontSize: '12px'
                                            }}
                                            wrapperClassName="dashboard-chart-tooltip"
                                        />
                                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                                        <Line 
                                            type="monotone" 
                                            dataKey="average_health_score" 
                                            stroke="#10b981" 
                                            strokeWidth={2}
                                            name="Health (%)" 
                                            dot={false}
                                        />
                                        <Line 
                                            type="monotone" 
                                            dataKey="average_failure_probability" 
                                            stroke="#ef4444" 
                                            strokeWidth={2}
                                            name="Failure Risk (%)" 
                                            dot={false}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </ChartPanel>
                        ) : (
                            <ChartPanel title="Health Trends" className="dashboard-chart-panel">
                                <div className="dashboard-cell-empty">No trend data available.</div>
                            </ChartPanel>
                        )}
                    </div>
                    <div className="dashboard-cell dashboard-cell-square dashboard-cell-square-top-right dashboard-cell-square-span">
                        <ChartPanel title="Status Distribution" className="dashboard-chart-panel">
                            <ResponsiveContainer width="100%" height="100%" minHeight={160}>
                                <PieChart>
                                    <Pie
                                        data={statusData}
                                        cx="50%"
                                        cy="50%"
                                        labelLine={false}
                                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                        outerRadius={80}
                                        fill="#8884d8"
                                        dataKey="value"
                                    >
                                        {statusData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip 
                                        contentStyle={{ 
                                            backgroundColor: '#ffffff', 
                                            border: '1px solid #cbd5e1',
                                            borderRadius: '8px'
                                        }}
                                    />
                                </PieChart>
                            </ResponsiveContainer>
                        </ChartPanel>
                    </div>

                    <div className="dashboard-cell dashboard-cell-square dashboard-cell-square-bottom">
                        {powerChartData.length > 0 ? (
                            <ChartPanel 
                                title="Power Output by Turbine"
                                subtitle="Average and max power (kW)"
                                className="dashboard-chart-panel"
                            >
                                <ResponsiveContainer width="100%" height="100%" minHeight={160}>
                                    <BarChart data={powerChartData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dashboard-chart-grid" />
                                        <XAxis 
                                            dataKey="name" 
                                            angle={-45} 
                                            textAnchor="end" 
                                            height={72}
                                            stroke="#64748b"
                                            tick={{ fontSize: 11 }}
                                            className="dashboard-chart-axis"
                                        />
                                        <YAxis 
                                            stroke="#64748b"
                                            tick={{ fontSize: 11 }}
                                            label={{ value: 'kW', angle: -90, position: 'insideLeft', style: { fill: '#64748b', fontSize: 11 } }}
                                            className="dashboard-chart-axis"
                                        />
                                        <Tooltip 
                                            contentStyle={{ 
                                                backgroundColor: '#ffffff', 
                                                border: '1px solid #cbd5e1',
                                                borderRadius: '8px'
                                            }}
                                        />
                                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                                        <Bar dataKey="average" fill="#0066cc" name="Avg (kW)" />
                                        <Bar dataKey="max" fill="#00a8e8" name="Max (kW)" />
                                    </BarChart>
                                </ResponsiveContainer>
                            </ChartPanel>
                        ) : (
                            <ChartPanel title="Power Output by Turbine" className="dashboard-chart-panel">
                                <div className="dashboard-cell-empty">No power data available.</div>
                            </ChartPanel>
                        )}
                    </div>

                    <div className="dashboard-cell dashboard-cell-big">
                        <div className="dashboard-summary-panel dashboard-big-detail">
                            <h3>System Status Summary</h3>
                            <div className="summary-content">
                                {stats.totalTurbines === 0 ? (
                                    <p>No turbines available.</p>
                                ) : (
                                    <>
                                        <div className="summary-item">
                                            <span className="summary-label">System Availability:</span>
                                            <span className="summary-value">{stats.healthPercentage}%</span>
                                            <span className="summary-detail">({stats.healthyCount} of {stats.totalTurbines} turbines operational)</span>
                                        </div>
                                        {stats.averageHealthScore !== null && (
                                            <div className="summary-item">
                                                <span className="summary-label">Average Health Score:</span>
                                                <span className="summary-value">{(stats.averageHealthScore * 100).toFixed(1)}%</span>
                                            </div>
                                        )}
                                        {(stats.warningCount > 0 || stats.criticalCount > 0) && (
                                            <div className="summary-item">
                                                <span className="summary-label">Turbines Requiring Attention:</span>
                                                <span className="summary-value warning">{stats.warningCount + stats.criticalCount}</span>
                                                <span className="summary-detail">({stats.warningCount} warning, {stats.criticalCount} critical)</span>
                                            </div>
                                        )}
                                        {stats.activeAlerts > 0 && (
                                            <div className={`summary-item ${stats.criticalAlerts > 0 ? 'alert-critical' : 'alert-warning'}`}>
                                                <span className="summary-label">Active Alerts:</span>
                                                <span className="summary-value">{stats.activeAlerts}</span>
                                                {stats.criticalAlerts > 0 && (
                                                    <span className="summary-detail">({stats.criticalAlerts} critical - immediate action required)</span>
                                                )}
                                            </div>
                                        )}
                                        {stats.activeAlerts === 0 && (
                                            <div className="summary-item success">
                                                <span className="summary-label">Alert Status:</span>
                                                <span className="summary-value">All Clear</span>
                                                <span className="summary-detail">No active alerts</span>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}

export default Dashboard;
