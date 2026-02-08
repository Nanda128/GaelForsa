import React, { useState, useEffect, useCallback } from 'react';
import { ThemeProvider } from './contexts/ThemeContext';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import MapView from './components/MapView';
import Dashboard from './components/Dashboard';
import GridImpact from './components/GridImpact';
import EconomicsDashboard from './components/EconomicsDashboard';
import TurbineDetails from './components/TurbineDetails';
import LoadingOverlay from './components/LoadingOverlay';
import ErrorMessage from './components/ErrorMessage';
import { useTurbines } from './hooks/useTurbines';
import { VIEWS, INITIAL_STATUS_FILTERS, REFRESH_INTERVAL_MS } from './constants/views';

function App() {
    const { turbines, loading, error, refreshTurbines } = useTurbines();
    const [currentView, setCurrentView] = useState(VIEWS.DASHBOARD);
const [isAppSidebarOpen, setIsAppSidebarOpen] = useState(true);
    const [selectedTurbine, setSelectedTurbine] = useState(null);
    const [showTurbineDetails, setShowTurbineDetails] = useState(false);
    const [statusFilters, setStatusFilters] = useState(INITIAL_STATUS_FILTERS);
    const [errorMessage, setErrorMessage] = useState(null);

    const handleTurbineClick = useCallback((turbine) => {
        setSelectedTurbine(turbine);
        setShowTurbineDetails(true);
    }, []);

    const handleCloseTurbineDetails = useCallback(() => {
        setShowTurbineDetails(false);
        setSelectedTurbine(null);
    }, []);

    const handleFilterChange = useCallback((filters) => {
        setStatusFilters(filters);
    }, []);

    const handleCloseError = useCallback(() => {
        setErrorMessage(null);
    }, []);

    const handleViewChange = useCallback((view) => {
        setCurrentView(view);
    }, []);

    useEffect(() => {
        if (error) {
            setErrorMessage(error);
        }
    }, [error]);

    useEffect(() => {
        // const interval = setInterval(() => {
        //     console.log('Auto-refreshing turbines...');
        //     refreshTurbines();
        // }, REFRESH_INTERVAL_MS);
        const interval = setInterval(refreshTurbines, REFRESH_INTERVAL_MS);
        return () => clearInterval(interval);
    }, [refreshTurbines]);

    const renderView = () => {
        switch (currentView) {
            case VIEWS.MAP:
                return (
                    <>
                        <Sidebar
                            turbines={turbines}
                            statusFilters={statusFilters}
                            onFilterChange={handleFilterChange}
                            onTurbineClick={handleTurbineClick}
                        />
                        <MapView
                            turbines={turbines}
                            statusFilters={statusFilters}
                            onTurbineClick={handleTurbineClick}
                        />
                    </>
                );
            case VIEWS.DASHBOARD:
                return (
                    <Dashboard
                        turbines={turbines}
                        onGridImpactClick={() => handleViewChange(VIEWS.GRID_IMPACT)}
                        onEconomicsClick={() => handleViewChange(VIEWS.ECONOMICS)}
                    />
                );
            case VIEWS.GRID_IMPACT:
                return (
                    <GridImpact
                        onBack={() => handleViewChange(VIEWS.DASHBOARD)}
                    />
                );
            case VIEWS.ECONOMICS:
                return (
                    <EconomicsDashboard
                        onBack={() => handleViewChange(VIEWS.DASHBOARD)}
                    />
                );
            default:
                return null;
        }
    };

    return (
        <ThemeProvider>
            <div className="app">
                <LoadingOverlay show={loading} />
                <ErrorMessage 
                    message={errorMessage} 
                    onClose={handleCloseError}
                />
                <Header
                    turbines={turbines}
                    onRefresh={refreshTurbines}
                    onDashboardClick={() => handleViewChange(VIEWS.DASHBOARD)}
                    onMapClick={() => handleViewChange(VIEWS.MAP)}
                    currentView={currentView}
                    onToggleSidebar={() => setIsAppSidebarOpen((o) => !o)}
                    isSidebarOpen={isAppSidebarOpen}
                />
                <main className="app-content">
                    {renderView()}
                </main>
                {showTurbineDetails && selectedTurbine && (
                    <TurbineDetails
                        turbine={selectedTurbine}
                        onClose={handleCloseTurbineDetails}
                    />
                )}
            </div>
        </ThemeProvider>
    );
}

export default App;
