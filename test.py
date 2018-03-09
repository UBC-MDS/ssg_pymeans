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
        return pymeans.fit(K=3)

    def test_type(self, result):
        assert isinstance(result, dict)

    def test_length(self, result):
        assert len(result) == 3

    def test_clustering_type(self, result):
        assert isinstance(result['data'], pd.DataFrame)

    def test_wss_type(self, result):
        assert isinstance(result['tot_withinss'], float)

    def test_wss_positive(self, result):
        assert result['tot_withinss'] >= 0

    def test_centriod_type(self, result):
        assert isinstance(result['centroids'], pd.DataFrame)

class Test_predict:
    @pytest.fixture
    def pymeans(self):
        test_data = pd.DataFrame({
            'x1': pd.Series([1,2,3]),
            'x2': pd.Series([4,5,6]),
            # 'cluster': pd.Series([1,2,3])
        })
        return Pymeans(data=test_data)

    @pytest.fixture
    def output(self, pymeans):
        test_data = pd.DataFrame({
            'x1': pd.Series([1.2,2.2,3.1]),
            'x2': pd.Series([4.1,5.1,6]),
            # 'cluster': pd.Series([1,2,3])
        })
        fit_results = pymeans.fit(K=2)
        return pymeans.predict(test_data, fit_results['centroids'])

    def test_shape(self, pymeans, output):
        assert pymeans.data.shape[0] == output.shape[0]

    def test_output_type(self, output):
        assert np.max(output[:, output.shape[1]-1])<4

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
