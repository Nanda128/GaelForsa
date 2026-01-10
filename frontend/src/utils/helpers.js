import { STATUS_LABELS, SEVERITY_LABELS } from '../constants/status';

export function formatDateTime(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString('en-IE', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

export function getStatusLabel(status) {
    return STATUS_LABELS[status] || 'Unknown';
}

export function getSeverityLabel(severity) {
    return SEVERITY_LABELS[severity] || 'Unknown';
}

export function formatHealthScore(score) {
    if (score === null || score === undefined) return 'N/A';
    return `${(score * 100).toFixed(1)}%`;
}

export function formatFailureProbability(probability) {
    if (probability === null || probability === undefined) return 'N/A';
    return `${(probability * 100).toFixed(2)}%`;
}

export function calculatePowerOutput(logs) {
    if (!logs || logs.length === 0) {
        return { total: 0, average: 0, max: 0, min: 0, count: 0 };
    }
    
    const powerValues = logs
        .map(log => log.on_turbine_readings?.[0]?.power_output)
        .filter(val => val != null && !isNaN(val));
    // .filter(val => typeof val === 'number' && val >= 0); // tried this but null check is enough
    
    if (powerValues.length === 0) {
        return { total: 0, average: 0, max: 0, min: 0, count: 0 };
    }
    
    const total = powerValues.reduce((sum, val) => sum + val, 0);
    const average = total / powerValues.length;
    // const sorted = [...powerValues].sort((a, b) => a - b);
    const max = Math.max(...powerValues);
    const min = Math.min(...powerValues);
    
    return { total, average, max, min, count: powerValues.length };
}
