import pytest
from ssg_pymeans import Pymeans, InvalidInput
import numpy as np
import pandas as pd

class Test_constructor:
    def test_input_type(self):
        test_data = np.ones(4)
        with pytest.raises(InvalidInput):
            pymeans = Pymeans(test_data)

class Test_init_cent:
    @pytest.fixture
    def pymeans(self):
        return Pymeans()

    @pytest.mark.parametrize("k,method", [
        (2, 1000),
        (0, 'random'),
        (100000000, 'kmpp'),
        (None, 'kmpp'),
        (2, 'fancy_method'),
        ('k=10', 'random')
    ])
    def test_input(self, pymeans, k, method):
        with pytest.raises(InvalidInput):
            pymeans.init_cent(k, method)

class Test_fit:
    @pytest.fixture
    def result(self):
        pymeans = Pymeans()
        return pymeans.fit()

    def test_type(self, result):
        assert type(result) == "dictionary"

    def test_length(self, result):
        assert len(result) == 3

    def test_clustering_type(self, result):
        assert isinstance(result[clustering], pd.DataFrame)

    def test_wss_type(self, result):
        assert type(result[wss]) == "float"

    def test_wss_positive(self, result):
        assert result[wss] >= 0

    def test_centriod_type(self, result):
        assert isinstance(result[centroid], pd.DataFrame)

class Test_predict:
    @pytest.fixture
    def pymeans(self):
        test_data = pd.DataFrame({
            'x1': pd.Series([1,2,3]),
            'x2': pd.Series([4,5,6]),
            'cluster': pd.Series([1,2,3])
        })
        return Pymeans(data=test_data)

    def output(self):
        pymeans = Pymeans()
        return pymeans.predict()

    def test_lenght(self, output):
        assert len(pymeans.data.rows) == len(output[:,-1])

    def test_output_type(self, output):
        assert max(output[:,-1])<4

class Test_kmplot:
    @pytest.fixture
    def pymeans(self):
        test_data = pd.DataFrame({
            'x1': pd.Series([1,2,3]),
            'x2': pd.Series([4,5,6]),
            'cluster': pd.Series([1,2,3])
        })
        return Pymeans(data=test_data)

    def test_kmplot(self, pymeans):
        lines = pymeans.kmplot()
        x_plot, y_plot = lines[0].get_data()
        # check if data in the plot match the input data
        np.testing.assert_array_equal(x_plot, pymeans.data.iloc[:,0].values)
        np.testing.assert_array_equal(y_plot, pymeans.data.iloc[:,1].values)

    @pytest.mark.parametrize("test_data", [
        pd.DataFrame({
            'x1': pd.Series([]),
            'x2': pd.Series([])
        }),
        pd.DataFrame({
            'x1': pd.Series([1,2,3]),
            'x2': pd.Series([4,5,6])
        })
    ])
    def test_kmplot_input(self, test_data):
        pymeans_test = Pymeans(data=test_data)
        with pytest.raises(InvalidInput):
            fig = pymeans_test.kmplot()

class Test_kmplot_grid:
    @pytest.fixture
    def result(self):
        pymeans = Pymeans()
        return pymeans.kmplot()

    def test_grid_shape(self, result):
        assert result.shape[0] == result.shape[1]

    def test_grid_col(self, result):
        assert result.shape[1] == test_data.shape[1]-1
