import csv
import json
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.db.models import Q, Avg, Count
from core.models import HealthPrediction, MaintenanceEvent, Turbine


class Command(BaseCommand):
    help = 'Validate ML model predictions against historical data using maintenance events as ground truth'

    def add_arguments(self, parser):
        parser.add_argument(
            '--start-date',
            type=str,
            help='Start date for validation (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            help='End date for validation (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--turbines',
            type=str,
            help='Comma-separated list of turbine numbers to filter (e.g., "1,2,3")'
        )
        parser.add_argument(
            '--fault-threshold',
            type=float,
            default=0.5,
            help='Threshold for failure probability to consider as predicted fault (default: 0.5)'
        )
        parser.add_argument(
            '--output-format',
            type=str,
            choices=['console', 'csv', 'json'],
            default='console',
            help='Output format for validation report (default: console)'
        )
        parser.add_argument(
            '--output-file',
            type=str,
            help='Output file path for CSV/JSON formats'
        )

    def handle(self, *args, **options):
        # Parse arguments
        start_date = None
        end_date = None
        if options['start_date']:
            start_date = datetime.strptime(options['start_date'], '%Y-%m-%d').date()
        if options['end_date']:
            end_date = datetime.strptime(options['end_date'], '%Y-%m-%d').date()

        turbine_ids = None
        if options['turbines']:
            turbine_ids = [int(x.strip()) for x in options['turbines'].split(',')]

        fault_threshold = options['fault_threshold']
        output_format = options['output_format']
        output_file = options['output_file']

        # Query predictions
        predictions = HealthPrediction.objects.select_related('turbine', 'log').order_by('timestamp')

        if start_date or end_date:
            date_filter = Q()
            if start_date:
                date_filter &= Q(timestamp__date__gte=start_date)
            if end_date:
                date_filter &= Q(timestamp__date__lte=end_date)
            predictions = predictions.filter(date_filter)

        if turbine_ids:
            turbine_names = [f"Kelmarsh {tid}" for tid in turbine_ids]
            predictions = predictions.filter(turbine__name__in=turbine_names)

        total_predictions = predictions.count()
        if total_predictions == 0:
            self.stdout.write(self.style.WARNING('No predictions found matching the criteria'))
            return

        self.stdout.write(f'Validating {total_predictions} predictions...')

        # Calculate metrics
        results = self.calculate_metrics(predictions, fault_threshold)

        # Generate report
        report = self.generate_report(results)

        # Output
        if output_format == 'console':
            self.output_console(report)
        elif output_format == 'csv':
            self.output_csv(report, output_file)
        elif output_format == 'json':
            self.output_json(report, output_file)

    def calculate_metrics(self, predictions, fault_threshold):
        results = {
            'overall': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'mae_health': 0, 'count': 0},
            'by_turbine': {},
            'by_year': {},
            'confusion_matrix': {'predicted_fault': {'actual_fault': 0, 'actual_no_fault': 0},
                                'predicted_no_fault': {'actual_fault': 0, 'actual_no_fault': 0}},
            'roc_data': []
        }

        for prediction in predictions:
            turbine = prediction.turbine
            year = prediction.timestamp.year

            # Initialize turbine and year dicts
            if turbine.name not in results['by_turbine']:
                results['by_turbine'][turbine.name] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'mae_health': 0, 'count': 0}
            if year not in results['by_year']:
                results['by_year'][year] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'mae_health': 0, 'count': 0}

            # Determine predicted fault
            predicted_fault = prediction.failure_probability >= fault_threshold

            # Determine actual fault: check for maintenance event of type 'failure' within predicted window or 30 days after
            actual_fault = self.check_actual_fault(prediction)

            # Update confusion matrix
            if predicted_fault:
                if actual_fault:
                    results['confusion_matrix']['predicted_fault']['actual_fault'] += 1
                else:
                    results['confusion_matrix']['predicted_fault']['actual_no_fault'] += 1
            else:
                if actual_fault:
                    results['confusion_matrix']['predicted_no_fault']['actual_fault'] += 1
                else:
                    results['confusion_matrix']['predicted_no_fault']['actual_no_fault'] += 1

            # Update TP, FP, TN, FN
            if predicted_fault and actual_fault:
                tp = 1
                fp = 0
                tn = 0
                fn = 0
            elif predicted_fault and not actual_fault:
                tp = 0
                fp = 1
                tn = 0
                fn = 0
            elif not predicted_fault and actual_fault:
                tp = 0
                fp = 0
                tn = 0
                fn = 1
            else:
                tp = 0
                fp = 0
                tn = 1
                fn = 0

            results['overall']['tp'] += tp
            results['overall']['fp'] += fp
            results['overall']['tn'] += tn
            results['overall']['fn'] += fn
            results['overall']['count'] += 1

            results['by_turbine'][turbine.name]['tp'] += tp
            results['by_turbine'][turbine.name]['fp'] += fp
            results['by_turbine'][turbine.name]['tn'] += tn
            results['by_turbine'][turbine.name]['fn'] += fn
            results['by_turbine'][turbine.name]['count'] += 1

            results['by_year'][year]['tp'] += tp
            results['by_year'][year]['fp'] += fp
            results['by_year'][year]['tn'] += tn
            results['by_year'][year]['fn'] += fn
            results['by_year'][year]['count'] += 1

            # Health score MAE: derive actual health as 1 if no fault in next 30 days, else 0
            actual_health = 1.0 if not actual_fault else 0.0
            mae = abs(prediction.health_score - actual_health)
            results['overall']['mae_health'] += mae
            results['by_turbine'][turbine.name]['mae_health'] += mae
            results['by_year'][year]['mae_health'] += mae

            # For ROC, collect (false_positive_rate, true_positive_rate) but since we have one threshold, collect probabilities
            results['roc_data'].append({
                'failure_probability': prediction.failure_probability,
                'actual_fault': actual_fault
            })

        # Average MAE
        if results['overall']['count'] > 0:
            results['overall']['mae_health'] /= results['overall']['count']
        for turbine in results['by_turbine']:
            if results['by_turbine'][turbine]['count'] > 0:
                results['by_turbine'][turbine]['mae_health'] /= results['by_turbine'][turbine]['count']
        for year in results['by_year']:
            if results['by_year'][year]['count'] > 0:
                results['by_year'][year]['mae_health'] /= results['by_year'][year]['count']

        return results

    def check_actual_fault(self, prediction):
        # Check for maintenance event of type 'failure' within predicted window or 30 days after prediction
        window_start = prediction.timestamp
        window_end = prediction.predicted_failure_window_end or (prediction.timestamp + timedelta(days=30))

        maintenance_events = MaintenanceEvent.objects.filter(
            turbine=prediction.turbine,
            event_type='failure',
            start_time__gte=window_start,
            start_time__lte=window_end
        )
        return maintenance_events.exists()

    def generate_report(self, results):
        def calculate_metrics(tp, fp, tn, fn):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }

        report = {
            'overall': calculate_metrics(
                results['overall']['tp'],
                results['overall']['fp'],
                results['overall']['tn'],
                results['overall']['fn']
            ),
            'by_turbine': {},
            'by_year': {},
            'confusion_matrix': results['confusion_matrix'],
            'mae_health': results['overall']['mae_health'],
            'total_predictions': results['overall']['count']
        }

        for turbine, data in results['by_turbine'].items():
            report['by_turbine'][turbine] = calculate_metrics(data['tp'], data['fp'], data['tn'], data['fn'])
            report['by_turbine'][turbine]['mae_health'] = data['mae_health']
            report['by_turbine'][turbine]['count'] = data['count']

        for year, data in results['by_year'].items():
            report['by_year'][year] = calculate_metrics(data['tp'], data['fp'], data['tn'], data['fn'])
            report['by_year'][year]['mae_health'] = data['mae_health']
            report['by_year'][year]['count'] = data['count']

        # ROC data: sort by failure_probability descending
        roc_data = sorted(results['roc_data'], key=lambda x: x['failure_probability'], reverse=True)
        report['roc_data'] = roc_data

        return report

    def output_console(self, report):
        self.stdout.write(self.style.SUCCESS('Validation Report'))
        self.stdout.write('=' * 50)

        self.stdout.write(f"Total Predictions: {report['total_predictions']}")
        self.stdout.write(f"Mean Absolute Error (Health Score): {report['mae_health']:.4f}")
        self.stdout.write()

        overall = report['overall']
        self.stdout.write("Overall Performance:")
        self.stdout.write(f"  Precision: {overall['precision']:.4f}")
        self.stdout.write(f"  Recall: {overall['recall']:.4f}")
        self.stdout.write(f"  F1-Score: {overall['f1_score']:.4f}")
        self.stdout.write(f"  Accuracy: {overall['accuracy']:.4f}")
        self.stdout.write(f"  TP: {overall['tp']}, FP: {overall['fp']}, TN: {overall['tn']}, FN: {overall['fn']}")
        self.stdout.write()

        self.stdout.write("Confusion Matrix:")
        cm = report['confusion_matrix']
        self.stdout.write(f"  Predicted Fault | Actual Fault: {cm['predicted_fault']['actual_fault']}, Actual No Fault: {cm['predicted_fault']['actual_no_fault']}")
        self.stdout.write(f"  Predicted No Fault | Actual Fault: {cm['predicted_no_fault']['actual_fault']}, Actual No Fault: {cm['predicted_no_fault']['actual_no_fault']}")
        self.stdout.write()

        self.stdout.write("Performance by Turbine:")
        for turbine, data in report['by_turbine'].items():
            self.stdout.write(f"  {turbine}: F1={data['f1_score']:.4f}, MAE={data['mae_health']:.4f}, Count={data['count']}")
        self.stdout.write()

        self.stdout.write("Performance by Year:")
        for year, data in report['by_year'].items():
            self.stdout.write(f"  {year}: F1={data['f1_score']:.4f}, MAE={data['mae_health']:.4f}, Count={data['count']}")

    def output_csv(self, report, output_file):
        if not output_file:
            self.stdout.write(self.style.ERROR('Output file required for CSV format'))
            return

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Category', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'MAE_Health', 'TP', 'FP', 'TN', 'FN', 'Count'])

            # Overall
            overall = report['overall']
            writer.writerow(['Overall', overall['precision'], overall['recall'], overall['f1_score'], overall['accuracy'], report['mae_health'], overall['tp'], overall['fp'], overall['tn'], overall['fn'], report['total_predictions']])

            # By turbine
            for turbine, data in report['by_turbine'].items():
                writer.writerow([turbine, data['precision'], data['recall'], data['f1_score'], data['accuracy'], data['mae_health'], data['tp'], data['fp'], data['tn'], data['fn'], data['count']])

            # By year
            for year, data in report['by_year'].items():
                writer.writerow([str(year), data['precision'], data['recall'], data['f1_score'], data['accuracy'], data['mae_health'], data['tp'], data['fp'], data['tn'], data['fn'], data['count']])

        self.stdout.write(self.style.SUCCESS(f'Report saved to {output_file}'))

    def output_json(self, report, output_file):
        if output_file:
            with open(output_file, 'w') as jsonfile:
                json.dump(report, jsonfile, indent=2)
            self.stdout.write(self.style.SUCCESS(f'Report saved to {output_file}'))
        else:
            self.stdout.write(json.dumps(report, indent=2))