import { Route, Routes } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import ErrorBoundary from './components/common/ErrorBoundary'
import Dashboard from './pages/Dashboard'
import DistrictDetail from './pages/DistrictDetail'
import RegimeTransitions from './pages/RegimeTransitions'
import Verification from './pages/Verification'
import ModelInfo from './pages/ModelInfo'

/**
 * Main application routes wrapped in the AppShell (PRD Section 19.2)
 */
export default function App() {
  return (
    <AppShell>
      <ErrorBoundary name="Main Application View">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/districts" element={<DistrictDetail />} />
          <Route path="/districts/:districtId" element={<DistrictDetail />} />
          <Route path="/regime" element={<RegimeTransitions />} />
          <Route path="/verification" element={<Verification />} />
          <Route path="/model-info" element={<ModelInfo />} />
        </Routes>
      </ErrorBoundary>
    </AppShell>
  )
}

