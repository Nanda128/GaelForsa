// Sidebar component for turbine listing with search and filter functionality

import {escapeHtml} from '../utils/helpers.js';

/**
 * @typedef {import('./turbines.js').Turbine} Turbine
 */

/**
 * Sidebar state management
 */
let sidebarState = {
    collapsed: false, searchQuery: '', statusFilters: {
        green: true, yellow: true, red: true
    }, turbines: [], onTurbineClick: null, onFilterChange: null
};

/**
 * Get status label for display
 * @param {'green'|'yellow'|'red'} status - Turbine status
 * @returns {string} Human-readable status label
 */
function getStatusLabel(status) {
    const labels = {
        green: 'Healthy', yellow: 'Warning', red: 'Critical'
    };
    return labels[status] || 'Unknown';
}

/**
 * Filter turbines based on current search query and status filters
 * @returns {Turbine[]} Filtered list of turbines
 */
function getFilteredTurbines() {
    return sidebarState.turbines.filter(turbine => {
        if (!sidebarState.statusFilters[turbine.status]) {
            return false;
        }

        if (sidebarState.searchQuery) {
            const query = sidebarState.searchQuery.toLowerCase();
            const name = turbine.name.toLowerCase();
            return name.includes(query);
        }

        return true;
    });
}

/**
 * Render the turbine list in the sidebar
 */
function renderTurbineList() {
    const listContainer = document.getElementById('turbine-list');
    if (!listContainer) return;

    const filteredTurbines = getFilteredTurbines();

    if (filteredTurbines.length === 0) {
        listContainer.innerHTML = '<div class="no-results">No turbines found</div>';
        return;
    }

    listContainer.innerHTML = filteredTurbines
        .map(turbine => `
            <div class="turbine-item" data-turbine-id="${turbine.id}" tabindex="0">
                <span class="status-dot ${turbine.status}"></span>
                <div class="turbine-item-content">
                    <div class="turbine-item-name">${escapeHtml(turbine.name)}</div>
                    <div class="turbine-item-status">${getStatusLabel(turbine.status)}</div>
                </div>
            </div>
        `)
        .join('');

    const items = listContainer.querySelectorAll('.turbine-item');
    items.forEach(item => {
        const turbineId = parseInt(item.dataset.turbineId, 10);
        const turbine = sidebarState.turbines.find(t => t.id === turbineId);

        item.addEventListener('click', () => {
            if (turbine && sidebarState.onTurbineClick) {
                sidebarState.onTurbineClick(turbine);
            }
        });

        item.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (turbine && sidebarState.onTurbineClick) {
                    sidebarState.onTurbineClick(turbine);
                }
            }
        });
    });
}

/**
 * Toggle sidebar collapsed state
 */
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    sidebarState.collapsed = !sidebarState.collapsed;
    sidebar.classList.toggle('collapsed', sidebarState.collapsed);

    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 300);
}

/**
 * Handle search input changes
 * @param {Event} event - Input event
 */
function handleSearchInput(event) {
    sidebarState.searchQuery = event.target.value;
    renderTurbineList();
}

/**
 * Handle filter checkbox changes
 * @param {'green'|'yellow'|'red'} status - Status being toggled
 * @param {boolean} checked - New checkbox state
 */
function handleFilterChange(status, checked) {
    sidebarState.statusFilters[status] = checked;
    renderTurbineList();

    if (sidebarState.onFilterChange) {
        sidebarState.onFilterChange(sidebarState.statusFilters);
    }
}

/**
 * Initialize the sidebar with event listeners
 * @param {{onFilterChange: Mock<Procedure>}} options - Configuration options
 * @param {Function} [options.onTurbineClick] - Callback when turbine is clicked
 * @param {Function} [options.onFilterChange] - Callback when status filters change
 */
export function initSidebar(options = {}) {
    sidebarState.onTurbineClick = options.onTurbineClick || null;
    sidebarState.onFilterChange = options.onFilterChange || null;

    const toggleBtn = document.getElementById('sidebar-toggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleSidebar);
    }

    const searchInput = document.getElementById('turbine-search');
    if (searchInput) {
        searchInput.addEventListener('input', handleSearchInput);
    }

    ['green', 'yellow', 'red'].forEach(status => {
        const checkbox = document.getElementById(`filter-${status}`);
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                handleFilterChange(status, e.target.checked);
            });
        }
    });
}

/**
 * Update the sidebar with turbine data
 * @param {Turbine[]} turbines - Array of turbine objects
 */
export function updateSidebarTurbines(turbines) {
    sidebarState.turbines = turbines;
    renderTurbineList();
}

/**
 * Get current status filters
 * @returns {Object} Current filter state
 */
export function getStatusFilters() {
    return {...sidebarState.statusFilters};
}

/**
 * Reset sidebar state to defaults (for testing)
 */
export function resetSidebarState() {
    sidebarState = {
        collapsed: false, searchQuery: '', statusFilters: {
            green: true, yellow: true, red: true
        }, turbines: [], onTurbineClick: null, onFilterChange: null
    };
}

