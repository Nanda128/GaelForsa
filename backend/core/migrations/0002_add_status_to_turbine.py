from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='turbine',
            name='status',
            field=models.CharField(choices=[('green', 'Healthy'), ('yellow', 'Warning'), ('red', 'Critical')], default='green', max_length=10),
        ),
    ]
