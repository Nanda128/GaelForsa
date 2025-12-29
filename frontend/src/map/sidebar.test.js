import {describe, it, expect, beforeEach, afterEach, vi} from 'vitest';
import {initSidebar, updateSidebarTurbines, getStatusFilters, resetSidebarState} from './sidebar.js';

// Mock turbine data
const mockTurbines = [{id: 1, name: 'Turbine Alpha', status: 'green', lat: 53.0, lng: -9.0}, {
    id: 2,
    name: 'Turbine Beta',
    status: 'yellow',
    lat: 53.1,
    lng: -9.1
}, {id: 3, name: 'Turbine Gamma', status: 'red', lat: 53.2, lng: -9.2}, {
    id: 4,
    name: 'Delta Wind Farm',
    status: 'green',
    lat: 53.3,
    lng: -9.3
}, {id: 5, name: 'Echo Generator', status: 'yellow', lat: 53.4, lng: -9.4}];

// mock DOM setup
function setupDOM() {
    document.body.innerHTML = `
        <div id="sidebar">
            <button id="sidebar-toggle">Toggle</button>
            <input type="text" id="turbine-search" placeholder="Search turbines..." />
            <div class="filter-controls">
                <label>
                    <input type="checkbox" id="filter-green" checked />
                    Healthy
                </label>
                <label>
                    <input type="checkbox" id="filter-yellow" checked />
                    Warning
                </label>
                <label>
                    <input type="checkbox" id="filter-red" checked />
                    Critical
                </label>
            </div>
            <div id="turbine-list"></div>
        </div>
    `;
}

function cleanupDOM() {
    document.body.innerHTML = '';
}

describe('Sidebar Module', () => {
    beforeEach(() => {
        resetSidebarState();
        setupDOM();
    });

    afterEach(() => {
        cleanupDOM();
    });

    describe('initSidebar', () => {
        it('should initialize without errors', () => {
            expect(() => initSidebar()).not.toThrow();
        });

        it('should accept onTurbineClick callback', () => {
            const onTurbineClick = vi.fn();
            expect(() => initSidebar({onTurbineClick})).not.toThrow();
        });

        it('should accept onFilterChange callback', () => {
            const onFilterChange = vi.fn();
            expect(() => initSidebar({onFilterChange})).not.toThrow();
        });

        it('should handle missing DOM elements gracefully', () => {
            cleanupDOM();
            expect(() => initSidebar()).not.toThrow();
        });
    });

    describe('updateSidebarTurbines', () => {
        beforeEach(() => {
            initSidebar();
        });

        it('should render turbine list', () => {
            updateSidebarTurbines(mockTurbines);
            const list = document.getElementById('turbine-list');
            expect(list.children.length).toBe(mockTurbines.length);
        });

        it('should display turbine names', () => {
            updateSidebarTurbines(mockTurbines);
            const list = document.getElementById('turbine-list');
            expect(list.innerHTML).toContain('Turbine Alpha');
            expect(list.innerHTML).toContain('Turbine Beta');
            expect(list.innerHTML).toContain('Turbine Gamma');
        });

        it('should display status dots with correct classes', () => {
            updateSidebarTurbines(mockTurbines);
            const dots = document.querySelectorAll('.status-dot');
            expect(dots.length).toBe(mockTurbines.length);

            const greenDots = document.querySelectorAll('.status-dot.green');
            const yellowDots = document.querySelectorAll('.status-dot.yellow');
            const redDots = document.querySelectorAll('.status-dot.red');

            expect(greenDots.length).toBe(2);
            expect(yellowDots.length).toBe(2);
            expect(redDots.length).toBe(1);
        });

        it('should display human-readable status labels', () => {
            updateSidebarTurbines(mockTurbines);
            const list = document.getElementById('turbine-list');
            expect(list.innerHTML).toContain('Healthy');
            expect(list.innerHTML).toContain('Warning');
            expect(list.innerHTML).toContain('Critical');
        });

        it('should show no results message for empty array', () => {
            updateSidebarTurbines([]);
            const list = document.getElementById('turbine-list');
            expect(list.innerHTML).toContain('No turbines found');
        });

        it('should set data-turbine-id attributes', () => {
            updateSidebarTurbines(mockTurbines);
            const items = document.querySelectorAll('.turbine-item');
            items.forEach((item, index) => {
                expect(item.dataset.turbineId).toBe(String(mockTurbines[index].id));
            });
        });

        it('should make turbine items focusable with tabindex', () => {
            updateSidebarTurbines(mockTurbines);
            const items = document.querySelectorAll('.turbine-item');
            items.forEach(item => {
                expect(item.getAttribute('tabindex')).toBe('0');
            });
        });
    });

    describe('Search functionality', () => {
        beforeEach(() => {
            initSidebar();
            updateSidebarTurbines(mockTurbines);
        });

        it('should filter turbines by search query', () => {
            const searchInput = document.getElementById('turbine-search');
            searchInput.value = 'Alpha';
            searchInput.dispatchEvent(new Event('input'));

            const items = document.querySelectorAll('.turbine-item');
            expect(items.length).toBe(1);
            expect(items[0].innerHTML).toContain('Turbine Alpha');
        });

        it('should be case-insensitive', () => {
            const searchInput = document.getElementById('turbine-search');
            searchInput.value = 'alpha';
            searchInput.dispatchEvent(new Event('input'));

            const items = document.querySelectorAll('.turbine-item');
            expect(items.length).toBe(1);
        });

        it('should search partial matches', () => {
            const searchInput = document.getElementById('turbine-search');
            searchInput.value = 'Turb';
            searchInput.dispatchEvent(new Event('input'));

            const items = document.querySelectorAll('.turbine-item');
            expect(items.length).toBe(3);
        });

        it('should show no results when no match found', () => {
            const searchInput = document.getElementById('turbine-search');
            searchInput.value = 'NonExistent';
            searchInput.dispatchEvent(new Event('input'));

            const list = document.getElementById('turbine-list');
            expect(list.innerHTML).toContain('No turbines found');
        });

        it('should show all turbines when search is cleared', () => {
            const searchInput = document.getElementById('turbine-search');
            searchInput.value = 'Alpha';
            searchInput.dispatchEvent(new Event('input'));

            searchInput.value = '';
            searchInput.dispatchEvent(new Event('input'));

            const items = document.querySelectorAll('.turbine-item');
            expect(items.length).toBe(mockTurbines.length);
        });
    });

    describe('Status filter functionality', () => {
        beforeEach(() => {
            initSidebar();
            updateSidebarTurbines(mockTurbines);
        });

        it('should filter out green status when unchecked', () => {
            const greenFilter = document.getElementById('filter-green');
            greenFilter.checked = false;
            greenFilter.dispatchEvent(new Event('change'));

            const items = document.querySelectorAll('.turbine-item');
            expect(items.length).toBe(3);
            expect(document.querySelectorAll('.status-dot.green').length).toBe(0);
        });

        it('should filter out yellow status when unchecked', () => {
            const yellowFilter = document.getElementById('filter-yellow');
            yellowFilter.checked = false;
            yellowFilter.dispatchEvent(new Event('change'));

            const items = document.querySelectorAll('.turbine-item');
            expect(items.length).toBe(3);
            expect(document.querySelectorAll('.status-dot.yellow').length).toBe(0);
        });

        it('should filter out red status when unchecked', () => {
            const redFilter = document.getElementById('filter-red');
            redFilter.checked = false;
            redFilter.dispatchEvent(new Event('change'));

            const items = document.querySelectorAll('.turbine-item');
            expect(items.length).toBe(4);
            expect(document.querySelectorAll('.status-dot.red').length).toBe(0);
        });

        it('should show no results when all filters unchecked', () => {
            const greenFilter = document.getElementById('filter-green');
            const yellowFilter = document.getElementById('filter-yellow');
            const redFilter = document.getElementById('filter-red');

            greenFilter.checked = false;
            greenFilter.dispatchEvent(new Event('change'));
            yellowFilter.checked = false;
            yellowFilter.dispatchEvent(new Event('change'));
            redFilter.checked = false;
            redFilter.dispatchEvent(new Event('change'));

            const list = document.getElementById('turbine-list');
            expect(list.innerHTML).toContain('No turbines found');
        });

        it('should call onFilterChange callback when filter changes', () => {
            const onFilterChange = vi.fn();
            initSidebar({onFilterChange});
            updateSidebarTurbines(mockTurbines);

            const greenFilter = document.getElementById('filter-green');
            greenFilter.checked = false;
            greenFilter.dispatchEvent(new Event('change'));

            expect(onFilterChange).toHaveBeenCalledWith({
                green: false, yellow: true, red: true
            });
        });
    });

    describe('getStatusFilters', () => {
        beforeEach(() => {
            initSidebar();
        });

        it('should return current filter state', () => {
            const filters = getStatusFilters();
            expect(filters).toHaveProperty('green');
            expect(filters).toHaveProperty('yellow');
            expect(filters).toHaveProperty('red');
        });

        it('should return all true by default', () => {
            const filters = getStatusFilters();
            expect(filters.green).toBe(true);
            expect(filters.yellow).toBe(true);
            expect(filters.red).toBe(true);
        });

        it('should reflect filter changes', () => {
            updateSidebarTurbines(mockTurbines);

            const greenFilter = document.getElementById('filter-green');
            greenFilter.checked = false;
            greenFilter.dispatchEvent(new Event('change'));

            const filters = getStatusFilters();
            expect(filters.green).toBe(false);
            expect(filters.yellow).toBe(true);
            expect(filters.red).toBe(true);
        });

        it('should return a copy, not the original object', () => {
            const filters1 = getStatusFilters();
            const filters2 = getStatusFilters();
            expect(filters1).not.toBe(filters2);
            expect(filters1).toEqual(filters2);
        });
    });

    describe('Sidebar toggle functionality', () => {
        beforeEach(() => {
            initSidebar();
        });

        it('should toggle collapsed class on sidebar', () => {
            const sidebar = document.getElementById('sidebar');
            const toggleBtn = document.getElementById('sidebar-toggle');

            expect(sidebar.classList.contains('collapsed')).toBe(false);

            toggleBtn.click();
            expect(sidebar.classList.contains('collapsed')).toBe(true);

            toggleBtn.click();
            expect(sidebar.classList.contains('collapsed')).toBe(false);
        });

        it('should dispatch resize event after toggle', async () => {
            vi.useFakeTimers();
            const resizeHandler = vi.fn();
            window.addEventListener('resize', resizeHandler);

            const toggleBtn = document.getElementById('sidebar-toggle');
            toggleBtn.click();

            vi.advanceTimersByTime(300);

            expect(resizeHandler).toHaveBeenCalled();

            window.removeEventListener('resize', resizeHandler);
            vi.useRealTimers();
        });
    });

    describe('Turbine click interaction', () => {
        it('should call onTurbineClick when turbine item is clicked', () => {
            const onTurbineClick = vi.fn();
            initSidebar({onTurbineClick});
            updateSidebarTurbines(mockTurbines);

            const firstItem = document.querySelector('.turbine-item');
            firstItem.click();

            expect(onTurbineClick).toHaveBeenCalledTimes(1);
            expect(onTurbineClick).toHaveBeenCalledWith(mockTurbines[0]);
        });

        it('should call onTurbineClick on Enter key press', () => {
            const onTurbineClick = vi.fn();
            initSidebar({onTurbineClick});
            updateSidebarTurbines(mockTurbines);

            const firstItem = document.querySelector('.turbine-item');
            const enterEvent = new KeyboardEvent('keydown', {key: 'Enter'});
            firstItem.dispatchEvent(enterEvent);

            expect(onTurbineClick).toHaveBeenCalledTimes(1);
            expect(onTurbineClick).toHaveBeenCalledWith(mockTurbines[0]);
        });

        it('should call onTurbineClick on Space key press', () => {
            const onTurbineClick = vi.fn();
            initSidebar({onTurbineClick});
            updateSidebarTurbines(mockTurbines);

            const firstItem = document.querySelector('.turbine-item');
            const spaceEvent = new KeyboardEvent('keydown', {key: ' '});
            firstItem.dispatchEvent(spaceEvent);

            expect(onTurbineClick).toHaveBeenCalledTimes(1);
            expect(onTurbineClick).toHaveBeenCalledWith(mockTurbines[0]);
        });

        it('should not call onTurbineClick on other key press', () => {
            const onTurbineClick = vi.fn();
            initSidebar({onTurbineClick});
            updateSidebarTurbines(mockTurbines);

            const firstItem = document.querySelector('.turbine-item');
            const tabEvent = new KeyboardEvent('keydown', {key: 'Tab'});
            firstItem.dispatchEvent(tabEvent);

            expect(onTurbineClick).not.toHaveBeenCalled();
        });
    });

    describe('Combined search and filter', () => {
        beforeEach(() => {
            initSidebar();
            updateSidebarTurbines(mockTurbines);
        });

        it('should apply both search and filter together', () => {
            const searchInput = document.getElementById('turbine-search');
            const greenFilter = document.getElementById('filter-green');

            searchInput.value = 'Turbine';
            searchInput.dispatchEvent(new Event('input'));

            greenFilter.checked = false;
            greenFilter.dispatchEvent(new Event('change'));

            const items = document.querySelectorAll('.turbine-item');
            expect(items.length).toBe(2);
        });

        it('should show no results when search and filter exclude all', () => {
            const searchInput = document.getElementById('turbine-search');
            const yellowFilter = document.getElementById('filter-yellow');
            const redFilter = document.getElementById('filter-red');

            searchInput.value = 'Alpha';
            searchInput.dispatchEvent(new Event('input'));

            yellowFilter.checked = true;
            redFilter.checked = true;
            const greenFilter = document.getElementById('filter-green');
            greenFilter.checked = false;
            greenFilter.dispatchEvent(new Event('change'));

            const list = document.getElementById('turbine-list');
            expect(list.innerHTML).toContain('No turbines found');
        });
    });

    describe('XSS protection', () => {
        beforeEach(() => {
            initSidebar();
        });

        it('should escape HTML in turbine names', () => {
            const maliciousTurbines = [{
                id: 1,
                name: '<script>alert("xss")</script>',
                status: 'green',
                lat: 53.0,
                lng: -9.0
            }];
            updateSidebarTurbines(maliciousTurbines);

            const list = document.getElementById('turbine-list');
            expect(list.innerHTML).not.toContain('<script>');
            expect(list.innerHTML).toContain('&lt;script&gt;');
        });
    });
});

