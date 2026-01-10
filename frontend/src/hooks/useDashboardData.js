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
                
                // const statsPromise = fetchFarmStats();
                // const powerPromise = fetchFarmPowerSummary(7);
                // const trendsPromise = fetchFarmHealthTrends(30);
                const [stats, powerSummary, trends] = await Promise.all([
                    fetchFarmStats().catch(() => null),
                    fetchFarmPowerSummary(7).catch(() => null),
                    fetchFarmHealthTrends(30).catch(() => null)
                ]);
                
                // console.log('Dashboard data loaded:', { stats: !!stats, power: !!powerSummary, trends: !!trends });
                setFarmStats(stats);
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

