export function exportToCSV(data, filename) {
    if (!data || data.length === 0) {
        console.error('No data to export');
        return;
    }

    const headers = Object.keys(data[0]);
    // const csvRows = [headers];
    const csvContent = [
        headers.join(','),
        ...data.map(row => 
            headers.map(header => {
                const value = row[header];
                if (value === null || value === undefined) return '';
                if (typeof value === 'object') return JSON.stringify(value);
                // return value.toString().replace(/"/g, '""');
                return `"${String(value).replace(/"/g, '""')}"`;
            }).join(',')
        )
    ].join('\n');
    // console.log('Exporting CSV:', filename, csvContent.length, 'chars');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename || 'export.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

export function exportTurbinesToCSV(turbines) {
    const data = turbines.map(turbine => ({
        'Turbine ID': turbine.id,
        'Name': turbine.name,
        'Status': turbine.status,
        'Latitude': turbine.latitude,
        'Longitude': turbine.longitude,
        'Created At': turbine.created_at,
        'Updated At': turbine.updated_at
    }));
    exportToCSV(data, `turbines_${new Date().toISOString().split('T')[0]}.csv`);
}

export function exportFarmStatsToCSV(stats) {
    const data = [{
        'Total Turbines': stats.turbines?.total || 0,
        'Healthy': stats.turbines?.by_status?.green || 0,
        'Warning': stats.turbines?.by_status?.yellow || 0,
        'Critical': stats.turbines?.by_status?.red || 0,
        'Health Percentage': stats.turbines?.health_percentage || 0,
        'Average Health Score': stats.health_metrics?.average_health_score || 0,
        'Average Failure Probability': stats.health_metrics?.average_failure_probability || 0,
        'Active Alerts': stats.alerts?.active || 0,
        'Critical Alerts': stats.alerts?.critical || 0,
        'Total Power (kW)': stats.power_output?.total_kw || 0,
        'Average Power (kW)': stats.power_output?.average_kw || 0,
        'Max Power (kW)': stats.power_output?.max_kw || 0,
        'Timestamp': stats.timestamp
    }];
    exportToCSV(data, `farm_stats_${new Date().toISOString().split('T')[0]}.csv`);
}

export function exportPowerDataToCSV(powerData) {
    const data = powerData.map(item => ({
        'Turbine Name': item.turbineName || item.turbine_name || 'Unknown',
        'Average Power (kW)': item.average || item.average_kw || 0,
        'Max Power (kW)': item.max || item.max_kw || 0,
        'Min Power (kW)': item.min || item.min_kw || 0,
        'Total Power (kW)': item.total || item.total_kw || 0,
        'Readings Count': item.count || item.readings_count || 0
    }));
    exportToCSV(data, `power_data_${new Date().toISOString().split('T')[0]}.csv`);
}

