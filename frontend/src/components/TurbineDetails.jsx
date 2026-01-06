import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { format, parseISO } from 'date-fns';
import {
    fetchTurbine,
    fetchTurbineAlerts,
    fetchTurbineHealthPredictions,
    fetchTurbineMaintenanceEvents,
    fetchTurbineLogs,
    acknowledgeAlert
} from '../utils/api';
import {
    getStatusLabel,
    getSeverityLabel,
    formatHealthScore,
    formatFailureProbability,
    formatDateTime,
    calculatePowerOutput
} from '../utils/helpers';

const TABS = {
    OVERVIEW: 'overview',
    POWER: 'power',
    HEALTH: 'health',
    ALERTS: 'alerts',
    MAINTENANCE: 'maintenance'
};

function TurbineDetails({ turbine: initialTurbine, onClose }) {
    const [activeTab, setActiveTab] = useState(TABS.OVERVIEW);
    const [turbine, setTurbine] = useState(initialTurbine);
    const [alerts, setAlerts] = useState([]);
    const [predictions, setPredictions] = useState([]);
    const [events, setEvents] = useState([]);
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function loadData() {
            try {
                setLoading(true);
                // Loading all at once - could lazy load per tab but this is simpler
                const [turbineData, alertsData, predictionsData, eventsData, logsData] = await Promise.all([
                    fetchTurbine(initialTurbine.id),
                    fetchTurbineAlerts(initialTurbine.id).catch(() => []),
                    fetchTurbineHealthPredictions(initialTurbine.id, 10).catch(() => []),
                    fetchTurbineMaintenanceEvents(initialTurbine.id, 10).catch(() => []),
                    fetchTurbineLogs(initialTurbine.id, 100).catch(() => [])
                ]);
                // console.log('Loaded data for turbine:', initialTurbine.id);
                setTurbine(turbineData);
                setAlerts(alertsData);
                setPredictions(predictionsData);
                setEvents(eventsData);
                setLogs(logsData);
            } catch (error) {
                console.error('Failed to load turbine details:', error);
            } finally {
                setLoading(false);
            }
        }
        loadData();
    }, [initialTurbine.id]);

    const handleAcknowledgeAlert = async (alertId) => {
        try {
            await acknowledgeAlert(turbine.id, alertId);
            const updatedAlerts = await fetchTurbineAlerts(turbine.id);
            setAlerts(updatedAlerts);
        } catch (error) {
            console.error('Failed to acknowledge alert:', error);
        }
    };

    const powerChartData = useMemo(() => {
        // Filter out logs without readings
        return logs
            .filter(log => log.on_turbine_readings && log.on_turbine_readings.length > 0)
            .map(log => {
                const reading = log.on_turbine_readings[0];
                // const power = reading.power_output ?? 0;
                return {
                    time: format(parseISO(log.timestamp), 'HH:mm'),
                    power: reading.power_output || 0,
                    rotorSpeed: reading.rotor_speed || 0
                };
            })
            .slice(-24)
            .reverse();
    }, [logs]);

    const healthChartData = useMemo(() => {
        return predictions.slice(0, 10).reverse().map(pred => ({
            time: format(parseISO(pred.timestamp), 'MMM dd HH:mm'),
            health: (pred.health_score * 100).toFixed(1),
            risk: (pred.failure_probability * 100).toFixed(1)
        }));
    }, [predictions]);

    const powerStats = useMemo(() => calculatePowerOutput(logs), [logs]);
    const activeAlerts = alerts.filter(a => !a.acknowledged);
    const latestPrediction = predictions[0];

    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };
        document.addEventListener('keydown', handleEscape);
        return () => document.removeEventListener('keydown', handleEscape);
    }, [onClose]);

    if (loading) {
        return (
            <div className={`turbine-details-panel active`}>
                <div className="turbine-details-loading">Loading...</div>
            </div>
        );
    }

    return (
        <div className={`turbine-details-panel active`}>
            <div className="turbine-details-header">
                <div>
                    <h2>{turbine.name}</h2>
                    <div className="turbine-details-subtitle">
                        {getStatusLabel(turbine.status)} • {activeAlerts.length} active alerts
                    </div>
                </div>
                <button 
                    className="turbine-details-close" 
                    onClick={onClose} 
                    aria-label="Close panel"
                    type="button"
                >
                    ×
                </button>
            </div>
            <div className="turbine-details-tabs">
                <button
                    className={`tab-btn ${activeTab === TABS.OVERVIEW ? 'active' : ''}`}
                    onClick={() => setActiveTab(TABS.OVERVIEW)}
                    type="button"
                >
                    Overview
                </button>
                <button
                    className={`tab-btn ${activeTab === TABS.POWER ? 'active' : ''}`}
                    onClick={() => setActiveTab(TABS.POWER)}
                    type="button"
                >
                    Power Output
                </button>
                <button
                    className={`tab-btn ${activeTab === TABS.HEALTH ? 'active' : ''}`}
                    onClick={() => setActiveTab(TABS.HEALTH)}
                    type="button"
                >
                    Health
                </button>
                <button
                    className={`tab-btn ${activeTab === TABS.ALERTS ? 'active' : ''}`}
                    onClick={() => setActiveTab(TABS.ALERTS)}
                    type="button"
                >
                    Alerts
                </button>
                <button
                    className={`tab-btn ${activeTab === TABS.MAINTENANCE ? 'active' : ''}`}
                    onClick={() => setActiveTab(TABS.MAINTENANCE)}
                    type="button"
                >
                    Maintenance
                </button>
            </div>
            <div className="turbine-details-content">
                {activeTab === TABS.OVERVIEW && (
                    <div>
                        <div className="overview-grid">
                            <div className="overview-card status-card">
                                <div className="card-label">Status</div>
                                <div className={`status-badge large ${turbine.status}`}>
                                    {getStatusLabel(turbine.status)}
                                </div>
                            </div>
                            <div className="overview-card">
                                <div className="card-label">Health Score</div>
                                <div className={`card-value ${latestPrediction ? (latestPrediction.health_score > 0.7 ? 'success' : latestPrediction.health_score > 0.4 ? 'warning' : 'danger') : ''}`}>
                                    {latestPrediction ? formatHealthScore(latestPrediction.health_score) : 'N/A'}
                                </div>
                            </div>
                            <div className="overview-card">
                                <div className="card-label">Failure Risk</div>
                                <div className={`card-value ${latestPrediction ? (latestPrediction.failure_probability < 0.2 ? 'success' : latestPrediction.failure_probability < 0.5 ? 'warning' : 'danger') : ''}`}>
                                    {latestPrediction ? formatFailureProbability(latestPrediction.failure_probability) : 'N/A'}
                                </div>
                            </div>
                            <div className="overview-card">
                                <div className="card-label">Active Alerts</div>
                                <div className={`card-value ${activeAlerts.length === 0 ? 'success' : activeAlerts.filter(a => a.severity === 'critical').length > 0 ? 'danger' : 'warning'}`}>
                                    {activeAlerts.length} {activeAlerts.filter(a => a.severity === 'critical').length > 0 ? `(${activeAlerts.filter(a => a.severity === 'critical').length} critical)` : ''}
                                </div>
                            </div>
                            <div className="overview-card">
                                <div className="card-label">Average Power</div>
                                <div className="card-value">
                                    {powerStats.average > 0 ? `${powerStats.average.toFixed(1)} kW` : 'N/A'}
                                </div>
                            </div>
                            <div className="overview-card">
                                <div className="card-label">Max Power</div>
                                <div className="card-value">
                                    {powerStats.max > 0 ? `${powerStats.max.toFixed(1)} kW` : 'N/A'}
                                </div>
                            </div>
                        </div>
                        <div className="overview-section">
                            <h4>Location</h4>
                            <div className="location-info">
                                <div>Lat: {turbine.latitude.toFixed(6)}</div>
                                <div>Lon: {turbine.longitude.toFixed(6)}</div>
                            </div>
                        </div>
                        <div className="overview-section">
                            <h4>Timestamps</h4>
                            <div className="timestamp-info">
                                <div><span>Last Updated:</span> {formatDateTime(turbine.updated_at)}</div>
                                <div><span>Created:</span> {formatDateTime(turbine.created_at)}</div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === TABS.POWER && (
                    <div>
                        {powerChartData.length > 0 ? (
                            <>
                                <div className="power-stats">
                                    <div className="stat-item">
                                        <div className="stat-label">Average</div>
                                        <div className="stat-value">{powerStats.average.toFixed(1)} kW</div>
                                    </div>
                                    <div className="stat-item">
                                        <div className="stat-label">Maximum</div>
                                        <div className="stat-value">{powerStats.max.toFixed(1)} kW</div>
                                    </div>
                                    <div className="stat-item">
                                        <div className="stat-label">Minimum</div>
                                        <div className="stat-value">{powerStats.min.toFixed(1)} kW</div>
                                    </div>
                                    <div className="stat-item">
                                        <div className="stat-label">Total</div>
                                        <div className="stat-value">{powerStats.total.toFixed(1)} kW</div>
                                    </div>
                                </div>
                                <div className="chart-container">
                                    <h4>Power Output Over Time</h4>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <AreaChart data={powerChartData}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="time" />
                                            <YAxis label={{ value: 'Power (kW)', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip />
                                            <Legend />
                                            <Area type="monotone" dataKey="power" stroke="#0066cc" fill="#0066cc" fillOpacity={0.6} name="Power Output (kW)" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="chart-container">
                                    <h4>Rotor Speed vs Power Output</h4>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <LineChart data={powerChartData}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="power" label={{ value: 'Power (kW)', position: 'insideBottom' }} />
                                            <YAxis label={{ value: 'Rotor Speed (rpm)', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip />
                                            <Legend />
                                            <Line type="monotone" dataKey="rotorSpeed" stroke="#00a8e8" name="Rotor Speed (rpm)" />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </>
                        ) : (
                            <div className="empty-state">
                                <div className="empty-title">No Power Data</div>
                                <div className="empty-message">No power output data available for this turbine.</div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === TABS.HEALTH && (
                    <div>
                        {predictions.length > 0 ? (
                            <>
                                <div className="health-summary">
                                    <div className="health-metric-card">
                                        <div className="metric-label">Current Health Score</div>
                                        <div className={`metric-value large ${latestPrediction.health_score > 0.7 ? 'success' : latestPrediction.health_score > 0.4 ? 'warning' : 'danger'}`}>
                                            {formatHealthScore(latestPrediction.health_score)}
                                        </div>
                                    </div>
                                    <div className="health-metric-card">
                                        <div className="metric-label">Failure Probability</div>
                                        <div className={`metric-value large ${latestPrediction.failure_probability < 0.2 ? 'success' : latestPrediction.failure_probability < 0.5 ? 'warning' : 'danger'}`}>
                                            {formatFailureProbability(latestPrediction.failure_probability)}
                                        </div>
                                    </div>
                                </div>
                                {latestPrediction.predicted_failure_window_start && (
                                    <div className="failure-window-alert">
                                        <div className="alert-icon">ALERT</div>
                                        <div>
                                            <div className="alert-title">Predicted Failure Window</div>
                                            <div className="alert-details">
                                                {formatDateTime(latestPrediction.predicted_failure_window_start)} - 
                                                {latestPrediction.predicted_failure_window_end ? formatDateTime(latestPrediction.predicted_failure_window_end) : 'Ongoing'}
                                            </div>
                                        </div>
                                    </div>
                                )}
                                {healthChartData.length > 0 && (
                                    <div className="chart-container">
                                        <h4>Health Trend</h4>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <LineChart data={healthChartData}>
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis dataKey="time" angle={-45} textAnchor="end" height={80} />
                                                <YAxis label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
                                                <Tooltip />
                                                <Legend />
                                                <Line type="monotone" dataKey="health" stroke="#10b981" name="Health Score (%)" />
                                                <Line type="monotone" dataKey="risk" stroke="#ef4444" name="Failure Risk (%)" />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="empty-state">
                                <div className="empty-title">No Health Data</div>
                                <div className="empty-message">No health predictions available for this turbine.</div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === TABS.ALERTS && (
                    <div>
                        {alerts.length === 0 ? (
                            <div className="empty-state">
                                <div className="empty-title">No Alerts</div>
                                <div className="empty-message">This turbine has no active or historical alerts.</div>
                            </div>
                        ) : (
                            <>
                                {activeAlerts.length > 0 && (
                                    <div className="alerts-section">
                                        <h4>Active Alerts ({activeAlerts.length})</h4>
                                        <div className="alerts-list-compact">
                                            {activeAlerts.map(alert => (
                                                <div key={alert.id} className={`alert-item-compact ${alert.severity}`}>
                                                    <div className="alert-header-compact">
                                                        <span className={`alert-severity-badge ${alert.severity}`}>
                                                            {getSeverityLabel(alert.severity)}
                                                        </span>
                                                        <span className="alert-time-compact">{formatDateTime(alert.created_at)}</span>
                                                    </div>
                                                    <div className="alert-message-compact">{alert.message}</div>
                                                    <button
                                                        className="alert-acknowledge-btn-small"
                                                        onClick={() => handleAcknowledgeAlert(alert.id)}
                                                        type="button"
                                                    >
                                                        Acknowledge
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                {alerts.filter(a => a.acknowledged).length > 0 && (
                                    <div className="alerts-section">
                                        <h4>Acknowledged ({alerts.filter(a => a.acknowledged).length})</h4>
                                        <div className="alerts-list-compact">
                                            {alerts.filter(a => a.acknowledged).map(alert => (
                                                <div key={alert.id} className={`alert-item-compact acknowledged ${alert.severity}`}>
                                                    <div className="alert-header-compact">
                                                        <span className={`alert-severity-badge ${alert.severity}`}>
                                                            {getSeverityLabel(alert.severity)}
                                                        </span>
                                                        <span className="alert-time-compact">{formatDateTime(alert.created_at)}</span>
                                                    </div>
                                                    <div className="alert-message-compact">{alert.message}</div>
                                                    <div className="alert-acknowledged-badge">
                                                        Acknowledged {alert.acknowledged_at ? formatDateTime(alert.acknowledged_at) : ''}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                )}

                {activeTab === TABS.MAINTENANCE && (
                    <div>
                        {events.length === 0 ? (
                            <div className="empty-state">
                                <div className="empty-title">No Maintenance Records</div>
                                <div className="empty-message">No maintenance events have been recorded for this turbine.</div>
                            </div>
                        ) : (
                            <div className="maintenance-list-compact">
                                {events.map(event => (
                                    <div key={event.id} className="maintenance-item-compact">
                                        <div className="maintenance-header-compact">
                                            <span className="event-type-badge">{event.event_type}</span>
                                            <span className="event-time-compact">{formatDateTime(event.start_time)}</span>
                                        </div>
                                        {event.description && (
                                            <div className="event-description-compact">{event.description}</div>
                                        )}
                                        <div className="event-details-compact">
                                            {event.end_time && (
                                                <span>Duration: {Math.round((new Date(event.end_time) - new Date(event.start_time)) / (1000 * 60 * 60))}h</span>
                                            )}
                                            {event.cost && (
                                                <span>Cost: €{parseFloat(event.cost).toFixed(2)}</span>
                                            )}
                                            {event.parts_replaced && (
                                                <span>Parts: {event.parts_replaced}</span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default TurbineDetails;
