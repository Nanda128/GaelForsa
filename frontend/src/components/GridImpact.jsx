import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useAsyncData } from '../hooks/useAsyncData';
import { fetchFarmGridImpact } from '../utils/api';
import StatCard from './shared/StatCard';
import ChartPanel from './shared/ChartPanel';
import PageHeader from './shared/PageHeader';
import Button from './shared/Button';
import LoadingState from './shared/LoadingState';
import ErrorState from './shared/ErrorState';

function GridImpact({ onBack }) {
    const { data, loading, error } = useAsyncData(fetchFarmGridImpact);

    if (loading) {
        return (
            <main className="dashboard-page">
                <LoadingState message="Loading grid impact data..." />
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

    if (!data) return null;

    // const operational = data.system_availability.operational_turbines;
    // const total = data.system_availability.total_turbines;
    const availabilityData = [
        { name: 'Operational', value: data.system_availability.operational_turbines, color: '#10b981' },
        { name: 'Downtime', value: data.system_availability.total_turbines - data.system_availability.operational_turbines, color: '#ef4444' }
    ];

    const powerData = [
        { name: 'Active Power', value: data.power_metrics.total_active_power_kw, color: '#0066cc' },
        { name: 'Reactive Power', value: data.power_metrics.estimated_reactive_power_kvar, color: '#00a8e8' }
    ];
    // console.log('Grid impact data:', { availability: availabilityData, power: powerData });

    const headerActions = onBack ? (
        <Button variant="secondary" onClick={onBack}>
            Back to Dashboard
        </Button>
    ) : null;

    return (
        <main className="dashboard-page">
            <PageHeader
                title="Grid Impact Analysis"
                subtitle="System availability and grid stability metrics"
                actions={headerActions}
            />
            <div className="dashboard-page-content">
                <div className="dashboard-stats-grid">
                    <StatCard
                        label="System Availability"
                        value={`${data.system_availability.percentage.toFixed(1)}%`}
                        unit={`${data.system_availability.operational_turbines} / ${data.system_availability.total_turbines} operational`}
                    />
                    <StatCard
                        label="Active Power"
                        value={`${data.power_metrics.total_active_power_kw.toFixed(0)}`}
                        unit="kW"
                    />
                    <StatCard
                        label="Reactive Power"
                        value={`${data.power_metrics.estimated_reactive_power_kvar.toFixed(0)}`}
                        unit="kVAR"
                    />
                    <StatCard
                        label="Critical Alerts"
                        value={data.grid_stability.critical_alerts_count}
                        unit="grid stability impact"
                        variant={data.grid_stability.critical_alerts_count > 0 ? 'danger' : 'default'}
                    />
                </div>

                <div className="dashboard-charts-grid">
                    <ChartPanel title="System Availability">
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={availabilityData}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                    outerRadius={100}
                                    fill="#8884d8"
                                    dataKey="value"
                                >
                                    {availabilityData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </ChartPanel>

                    <ChartPanel 
                        title="Power Distribution"
                        subtitle="Active vs Reactive Power"
                    >
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={powerData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis 
                                    dataKey="name" 
                                    stroke="#64748b"
                                    tick={{ fontSize: 11 }}
                                />
                                <YAxis 
                                    stroke="#64748b"
                                    tick={{ fontSize: 11 }}
                                />
                                <Tooltip 
                                    contentStyle={{ 
                                        backgroundColor: '#ffffff', 
                                        border: '1px solid #cbd5e1',
                                        borderRadius: '4px'
                                    }}
                                />
                                <Legend />
                                <Bar dataKey="value" fill="#0066cc" />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartPanel>
                </div>

                <div className="dashboard-summary-panel">
                    <h3>Grid Stability Metrics</h3>
                    <div className="summary-content">
                        <div className="summary-item">
                            <span className="summary-label">System Availability:</span>
                            <span className="summary-value">{data.system_availability.percentage.toFixed(1)}%</span>
                            <span className="summary-detail">({data.system_availability.operational_turbines} of {data.system_availability.total_turbines} turbines operational)</span>
                        </div>
                        <div className="summary-item">
                            <span className="summary-label">Total Apparent Power:</span>
                            <span className="summary-value">{data.power_metrics.apparent_power_kva.toFixed(0)} kVA</span>
                            <span className="summary-detail">(Active: {data.power_metrics.total_active_power_kw.toFixed(0)} kW, Reactive: {data.power_metrics.estimated_reactive_power_kvar.toFixed(0)} kVAR)</span>
                        </div>
                        {data.grid_stability.critical_alerts_count > 0 ? (
                            <div className="summary-item alert-critical">
                                <span className="summary-label">Critical Alerts:</span>
                                <span className="summary-value danger">{data.grid_stability.critical_alerts_count}</span>
                                <span className="summary-detail">May impact grid stability. Immediate attention recommended.</span>
                            </div>
                        ) : (
                            <div className="summary-item success">
                                <span className="summary-label">Alert Status:</span>
                                <span className="summary-value">All Clear</span>
                                <span className="summary-detail">No critical alerts. Grid stability maintained.</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
}

export default GridImpact;
