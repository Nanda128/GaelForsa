from rest_framework import serializers


class BaseSerializer(serializers.ModelSerializer):
    class Meta:
        abstract = True

