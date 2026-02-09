import { useState, useEffect } from 'react';
import { fetchFarmStats, fetchFarmPowerSummary, fetchFarmHealthTrends } from '../utils/api';

export function useDashboardData(turbines = []) {
    const [farmStats, setFarmStats] = useState(null);
    const [powerData, setPowerData] = useState([]);
    const [healthTrends, setHealthTrends] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function loadData() {
            try {
                setLoading(true);
                setError(null);

                const [stats, powerSummary, trends] = await Promise.all([
                    fetchFarmStats().catch(() => null),
                    fetchFarmPowerSummary(7).catch(() => null),
                    fetchFarmHealthTrends(30).catch(() => null)
                ]);

                if (stats) {
                    const statusCounts = stats.turbines_by_status || {};
                    const operational = statusCounts.operational || statusCounts.normal || 0;
                    const warning = statusCounts.warning || statusCounts.yellow || 0;
                    const critical = statusCounts.critical || statusCounts.red || 0;
                    const maintenance = statusCounts.maintenance || 0;

                    setFarmStats({
                        turbines: {
                            total: stats.total_turbines || 0,
                            by_status: {
                                green: operational,
                                yellow: warning,
                                red: critical,
                                maintenance: maintenance
                            },
                            health_percentage: stats.total_turbines > 0
                                ? (operational / stats.total_turbines) * 100
                                : 0
                        },
                        health_metrics: {
                            average_health_score: stats.average_health_score || 0
                        },
                        alerts: {
                            active: 0,
                            critical: 0
                        }
                    });
                }

                if (powerSummary) {
                    setPowerData(powerSummary.by_turbine || []);
                }

                if (trends) {
                    setHealthTrends(trends.trends || []);
                }
            } catch (err) {
                setError(err.message);
                console.error('Failed to load dashboard data:', err);
            } finally {
                setLoading(false);
            }
        }

        loadData();
    }, [turbines.length]);

    return {
        farmStats,
        powerData,
        healthTrends,
        loading,
        error
    };
}

