#!/usr/bin/env python3
"""
Comprehensive test script for validating the complete simulation pipeline.

This script tests:
1. Data loading (load_kelmarsh_data command)
2. Real-time simulation (simulate_realtime_feed command)
3. Validation (validate_predictions command)

It includes assertions, cleanup, logging, and error reporting.
"""
import os
import sys
import logging
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../GaelForsa/backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api.settings')
import django
django.setup()
from django.core.management import call_command
from django.db import connection
from core.models import Turbine, TurbineLog, OnTurbineReading, HealthPrediction, MaintenanceEvent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    handlers=[
        logging.FileHandler('test_simualtion_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimulationPipelineTest:
    """Test class for the simulation pipeline."""
    def __init__(self):
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'ai_model/5841834')
        self.backend_dir = os.path.join(os.path.dirname(__file__), '../GaelForsa/backend')

    def setup_test_enviorment(self):
        """Setup test environment."""
        logger.info("Setting up test enviorment")
        os.chdir(self.backend_dir)

    def cleanup_database(self):
        """Reset database between tests."""
        logger.info("Cleaning up database")
        try:
            HealthPrediction.objects.all().delete()
            OnTurbineReading.objects.all().delete()
            TurbineLog.objects.all().delete()
            MaintenanceEvent.object.all().delete()
            Turbine.objects.all().delete()
            if connection.vendor == 'postgresql':
                with connection.cursor() as cursor:
                    cursor.execute("SELECT setval(pg_get_serial_sequence('core_turbine','id'), 1, false);")
                    # Add other sequences as needed

            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
            raise

    def test_data_loading(self):
        """Test data loading component."""
        logger.info("Testing data loading...")

        try:
            # Use subset of data for testing (only 2016, turbine 1)
            call_command('load_kelmarsh_data', data_dir=self.test_data_dir, verbosity=1)

            # Assertions
            turbines_count = Turbine.objects.count()
            logs_count = TurbineLog.objects.count()
            readings_count = OnTurbineReading.objects.count()

            assert turbines_count > 0, "No turbines were created"
            assert logs_count > 0, "No turbine logs were created"
            assert readings_count > 0, "No readings were loaded"

            logger.info(f"Data loading successful: {turbines_count} turbines, {logs_count} logs, {readings_count} readings")

            return True

        except Exception as e:
            logger.error(f"Data loading test failed: {e}")
            return False

    def test_realtime_simulation(self):
        """Test real-time simulation component."""
        logger.info("Testing real-time simulation...")

        try:
            # Run simulation with limited data (first 100 readings, dry-run first for speed)
            initial_predictions = HealthPrediction.objects.count()

            # Use dry-run first to test without storing
            call_command('simulate_realtime_feed',
                        speed_multiplier=1000.0,  # Very fast
                        turbines='1',  # Only turbine 1
                        start_date='2016-01-01',
                        end_date='2016-01-02',  # Limited date range
                        verbosity=1)

            # Check that predictions were created
            final_predictions = HealthPrediction.objects.count()
            predictions_created = final_predictions - initial_predictions

            assert predictions_created > 0, "No predictions were generated"

            logger.info(f"Real-time simulation successful: {predictions_created} predictions created")

            return True

        except Exception as e:
            logger.error(f"Real-time simulation test failed: {e}")
            return False

    def test_validation(self):
        """Test validation component."""
        logger.info("Testing validation...")

        try:
            # Run validation on generated predictions
            predictions_count = HealthPrediction.objects.count()
            assert predictions_count > 0, "No predictions available for validation"

            # Capture output by redirecting stdout
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                call_command('validate_predictions',
                           start_date='2016-01-01',
                           end_date='2016-01-02',
                           turbines='1',
                           output_format='console',
                           verbosity=1)

                output = sys.stdout.getvalue()
                logger.info("Validation output:")
                logger.info(output)

                # Basic assertions on output
                assert 'Validation Report' in output, "Validation report not generated"
                assert 'Total Predictions:' in output, "Total predictions not reported"
                assert 'F1-Score:' in output, "F1-Score not calculated"

                logger.info("Validation successful: metrics calculated")

            finally:
                sys.stdout = old_stdout

            return True

        except Exception as e:
            logger.error(f"Validation test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all tests in sequence."""
        logger.info("Starting simulation pipeline tests...")

        self.setup_test_environment()

        test_results = []

        # Test 1: Data Loading
        self.cleanup_database()
        result1 = self.test_data_loading()
        test_results.append(('Data Loading', result1))

        # Test 2: Real-time Simulation
        result2 = self.test_realtime_simulation()
        test_results.append(('Real-time Simulation', result2))

        # Test 3: Validation
        result3 = self.test_validation()
        test_results.append(('Validation', result3))

        # Final cleanup
        self.cleanup_database()

        # Report results
        logger.info("\n" + "="*50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*50)

        all_passed = True
        for test_name, passed in test_results:
            status = "PASSED" if passed else "FAILED"
            logger.info(f"{test_name}: {status}")
            if not passed:
                all_passed = False

        logger.info("="*50)
        overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
        logger.info(overall_status)

        return all_passed


def main():
    """Main function."""
    try:
        tester = SimulationPipelineTest()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
