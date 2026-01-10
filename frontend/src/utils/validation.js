export function isValidTurbine(turbine) {
    if (!turbine) return false;
    if (!turbine.status) return false;
    if (turbine.latitude === undefined || turbine.latitude === null) return false;
    if (turbine.longitude === undefined || turbine.longitude === null) return false;
    
    const lat = parseFloat(turbine.latitude);
    const lon = parseFloat(turbine.longitude);
    
    if (isNaN(lat) || isNaN(lon)) return false;
    if (lat < -90 || lat > 90) return false;
    if (lon < -180 || lon > 180) return false;
    
    return true;
}

export function filterTurbinesByStatus(turbines, statusFilters) {
    if (!Array.isArray(turbines)) return [];
    
    return turbines.filter(turbine => {
        if (!isValidTurbine(turbine)) return false;
        return statusFilters[turbine.status] === true;
    });
}

