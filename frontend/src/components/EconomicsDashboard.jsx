import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useAsyncData } from '../hooks/useAsyncData';
import { fetchFarmEconomics } from '../utils/api';
import StatCard from './shared/StatCard';
import ChartPanel from './shared/ChartPanel';
import PageHeader from './shared/PageHeader';
import Button from './shared/Button';
import LoadingState from './shared/LoadingState';
import ErrorState from './shared/ErrorState';

function EconomicsDashboard({ onBack }) {
    const { data, loading, error } = useAsyncData(fetchFarmEconomics);

    if (loading) {
        return (
            <main className="dashboard-page">
                <LoadingState message="Loading economics data..." />
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

    // const revenue = data.revenue.total_revenue_eur;
    // const costs = data.costs.total_cost_eur;
    const revenueCostData = [
        { name: 'Revenue', value: data.revenue.total_revenue_eur, color: '#10b981' },
        { name: 'Maintenance', value: data.costs.maintenance_cost_eur, color: '#f59e0b' },
        { name: 'Downtime', value: data.costs.downtime_cost_eur, color: '#ef4444' }
    ];

    const savingsData = [
        { name: 'Preventive Savings', value: data.savings.estimated_preventive_savings_eur, color: '#10b981' },
        { name: 'Net Benefit', value: data.savings.net_benefit_eur, color: '#0066cc' }
    ];
    // console.log('Economics chart data:', revenueCostData, savingsData);

    const headerActions = onBack ? (
        <Button variant="secondary" onClick={onBack}>
            Back to Dashboard
        </Button>
    ) : null;

    return (
        <main className="dashboard-page">
            <PageHeader
                title="Economic Impact Analysis"
                subtitle="Revenue, costs, and ROI metrics"
                actions={headerActions}
            />
            <div className="dashboard-page-content">
                <div className="dashboard-stats-grid">
                    <StatCard
                        label="Total Revenue"
                        value={`€${data.revenue.total_revenue_eur.toLocaleString()}`}
                        unit={`${data.revenue.period_days} day period`}
                    />
                    <StatCard
                        label="ROI"
                        value={`${data.metrics.roi_percentage.toFixed(1)}%`}
                        unit="return on investment"
                        variant="success"
                    />
                    <StatCard
                        label="Total Costs"
                        value={`€${data.costs.total_cost_eur.toLocaleString()}`}
                        unit="operational expenses"
                        variant="warning"
                    />
                    <StatCard
                        label="Net Benefit"
                        value={`€${data.savings.net_benefit_eur.toLocaleString()}`}
                        unit="total benefit"
                        variant="success"
                    />
                </div>

                <div className="dashboard-charts-grid">
                    <ChartPanel 
                        title="Revenue vs Costs"
                        subtitle="Financial performance breakdown"
                    >
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={revenueCostData}>
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
                                    formatter={(value) => `€${value.toLocaleString()}`}
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

                    <ChartPanel 
                        title="Savings & Benefits"
                        subtitle="Preventive maintenance impact"
                    >
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={savingsData}>
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
                                    formatter={(value) => `€${value.toLocaleString()}`}
                                    contentStyle={{ 
                                        backgroundColor: '#ffffff', 
                                        border: '1px solid #cbd5e1',
                                        borderRadius: '4px'
                                    }}
                                />
                                <Legend />
                                <Bar dataKey="value" fill="#10b981" />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartPanel>
                </div>

                <div className="dashboard-summary-panel">
                    <h3>Economic Summary</h3>
                    <div className="summary-content">
                        <div className="summary-item">
                            <span className="summary-label">Energy Generated:</span>
                            <span className="summary-value">{data.revenue.total_energy_kwh.toLocaleString()} kWh</span>
                            <span className="summary-detail">over {data.revenue.period_days} days</span>
                        </div>
                        <div className="summary-item">
                            <span className="summary-label">Total Revenue:</span>
                            <span className="summary-value">€{data.revenue.total_revenue_eur.toLocaleString()}</span>
                        </div>
                        <div className="summary-item">
                            <span className="summary-label">Total Operational Costs:</span>
                            <span className="summary-value warning">€{data.costs.total_cost_eur.toLocaleString()}</span>
                            <span className="summary-detail">(Maintenance: €{data.costs.maintenance_cost_eur.toLocaleString()}, Downtime: €{data.costs.downtime_cost_eur.toLocaleString()})</span>
                        </div>
                        <div className="summary-item">
                            <span className="summary-label">Estimated Preventive Savings:</span>
                            <span className="summary-value success">€{data.savings.estimated_preventive_savings_eur.toLocaleString()}</span>
                        </div>
                        <div className={`summary-item ${data.metrics.roi_percentage > 0 ? 'success' : 'alert-warning'}`}>
                            <span className="summary-label">ROI:</span>
                            <span className="summary-value">{data.metrics.roi_percentage.toFixed(1)}%</span>
                            <span className="summary-detail">Cost per MWh: €{data.metrics.cost_per_mwh.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}

export default EconomicsDashboard;
