import { Compass } from 'lucide-react';
import { Link, Route, Routes } from 'react-router-dom';
import { EmptyState } from './components/ui';
import AppLayout from './layouts/AppLayout';
import DatasetExplorer from './pages/DatasetExplorer';
import Documentation from './pages/Documentation';
import Experiments from './pages/Experiments';
import ImageAnalysis from './pages/ImageAnalysis';
import ModelEvaluation from './pages/ModelEvaluation';
import ModelTraining from './pages/ModelTraining';
import NavigationSimulator from './pages/NavigationSimulator';
import Overview from './pages/Overview';
import Settings from './pages/Settings';
import TerrainIntelligence from './pages/TerrainIntelligence';

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Overview />} />
        <Route path="analysis" element={<ImageAnalysis />} />
        <Route path="dataset" element={<DatasetExplorer />} />
        <Route path="training" element={<ModelTraining />} />
        <Route path="evaluation" element={<ModelEvaluation />} />
        <Route path="terrain" element={<TerrainIntelligence />} />
        <Route path="navigation" element={<NavigationSimulator />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="docs" element={<Documentation />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={
          <EmptyState icon={Compass} title="Off the map" action={<Link to="/" className="btn-primary">Back to Overview</Link>}>
            This page does not exist.
          </EmptyState>
        } />
      </Route>
    </Routes>
  );
}
