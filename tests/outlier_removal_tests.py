import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from micromet.outlier_removal import detect_extreme_variations, clean_extreme_variations


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='H')
    np.random.seed(42)

    df = pd.DataFrame({
        'normal': np.random.normal(20, 5, len(dates)),
        'with_extremes': np.random.normal(20, 5, len(dates)),
        'with_nulls': np.random.normal(20, 5, len(dates)),
        'constant': np.ones(len(dates)) * 10
    }, index=dates)

    # Add known extreme values at specific hours
    df.loc['2024-01-05 12:00:00', 'with_extremes'] = 100
    df.loc['2024-01-07 12:00:00', 'with_extremes'] = -50

    # Add known null values at specific hours
    df.loc['2024-01-03 12:00:00', 'with_nulls'] = -9999
    df.loc['2024-01-06 12:00:00', 'with_nulls'] = -9999

    return df


class TestDetectExtremeVariations:
    """Test suite for detect_extreme_variations function."""

    def test_basic_functionality(self, sample_df):
        """Test basic function execution and return structure."""
        results = detect_extreme_variations(sample_df)

        assert isinstance(results, dict)
        assert all(key in results for key in ['variations', 'extreme_points', 'summary'])
        assert isinstance(results['variations'], pd.DataFrame)
        assert isinstance(results['extreme_points'], pd.DataFrame)
        assert isinstance(results['summary'], pd.DataFrame)

    def test_datetime_index_validation(self):
        """Test that function raises error for non-datetime index."""
        df = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        with pytest.raises(ValueError, match="DataFrame must have a datetime index"):
            detect_extreme_variations(df)

    def test_field_selection(self, sample_df):
        """Test field selection functionality."""
        # Test single field
        results = detect_extreme_variations(sample_df, fields='normal')
        assert len(results['summary']) == 1
        assert results['summary']['field'].iloc[0] == 'normal'

        # Test multiple fields
        results = detect_extreme_variations(sample_df, fields=['normal', 'with_extremes'])
        assert len(results['summary']) == 2
        assert all(field in results['summary']['field'].values
                   for field in ['normal', 'with_extremes'])

    def test_null_value_handling(self, sample_df):
        """Test handling of null values."""
        results = detect_extreme_variations(sample_df,
                                            fields='with_nulls',
                                            null_value=-9999)

        # Check that null values are properly excluded
        variations = results['variations']
        null_time = pd.Timestamp('2024-01-03 12:00:00')
        assert pd.isna(variations.loc[null_time, 'with_nulls_variation'])

    def test_extreme_detection(self, sample_df):
        """Test detection of known extreme values."""
        results = detect_extreme_variations(sample_df,
                                            fields='with_extremes',
                                            variation_threshold=3.0)

        extreme_points = results['extreme_points']
        extreme_time = pd.Timestamp('2024-01-05 12:00:00')
        assert extreme_points.loc[extreme_time, 'with_extremes_extreme']

    def test_constant_field(self, sample_df):
        """Test handling of constant fields."""
        results = detect_extreme_variations(sample_df, fields='constant')
        variations = results['variations']

        # All variations should be NaN for constant field
        assert variations['constant_variation'].isna().all()


class TestCleanExtremeVariations:
    """Test suite for clean_extreme_variations function."""

    def test_basic_functionality(self, sample_df):
        """Test basic function execution and return structure."""
        results = clean_extreme_variations(sample_df)

        assert isinstance(results, dict)
        assert all(key in results for key in ['cleaned_data', 'cleaning_summary', 'removed_points'])
        assert isinstance(results['cleaned_data'], pd.DataFrame)
        assert isinstance(results['cleaning_summary'], pd.DataFrame)
        assert isinstance(results['removed_points'], pd.DataFrame)

    def test_replacement_methods(self, sample_df):
        """Test different replacement methods."""
        extreme_time = pd.Timestamp('2024-01-05 12:00:00')

        # Test NaN replacement
        nan_results = clean_extreme_variations(sample_df,
                                               fields='with_extremes',
                                               replacement_method='nan')
        assert pd.isna(nan_results['cleaned_data'].loc[extreme_time, 'with_extremes'])

        # Test interpolation
        interp_results = clean_extreme_variations(sample_df,
                                                  fields='with_extremes',
                                                  replacement_method='interpolate')
        value = interp_results['cleaned_data'].loc[extreme_time, 'with_extremes']
        assert not pd.isna(value)
        assert abs(value - 100) > 1  # Should be significantly different from extreme value

        # Test mean replacement
        mean_results = clean_extreme_variations(sample_df,
                                                fields='with_extremes',
                                                replacement_method='mean')
        assert not pd.isna(mean_results['cleaned_data'].loc[extreme_time, 'with_extremes'])

    def test_invalid_replacement_method(self, sample_df):
        """Test invalid replacement method handling."""
        with pytest.raises(ValueError, match="replacement_method must be one of"):
            clean_extreme_variations(sample_df, replacement_method='invalid')

    def test_removed_points_tracking(self, sample_df):
        """Test tracking of removed points."""
        results = clean_extreme_variations(sample_df,
                                           fields='with_extremes',
                                           variation_threshold=2.0)  # Lower threshold to ensure detection
        removed = results['removed_points'].dropna(how='all')

        # Check if extreme values are in removed points
        extreme_time_1 = pd.Timestamp('2024-01-05 12:00:00')
        extreme_time_2 = pd.Timestamp('2024-01-07 12:00:00')
        assert abs(removed.loc[extreme_time_1, 'with_extremes'] - 100) < 0.1
        assert abs(removed.loc[extreme_time_2, 'with_extremes'] - (-50)) < 0.1

    def test_cleaning_summary(self, sample_df):
        """Test cleaning summary statistics."""
        results = clean_extreme_variations(sample_df, fields=['with_extremes', 'normal'])
        summary = results['cleaning_summary']

        assert len(summary) == 2
        assert all(col in summary.columns for col in
                   ['field', 'points_removed', 'percent_removed', 'replacement_method'])



if __name__ == "__main__":
    pytest.main([__file__])