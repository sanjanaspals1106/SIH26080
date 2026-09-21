import { NavLink, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import DistrictDetail from './pages/DistrictDetail'
import RegimeTransitions from './pages/RegimeTransitions'
import Verification from './pages/Verification'
import ModelInfo from './pages/ModelInfo'

// The five pages of PRD section 19.2. Each one is a placeholder that shows only its title.
export default function App() {
  return (
    <>
      <nav className="nav">
        <NavLink to="/" end>Dashboard</NavLink>
        <NavLink to="/districts">District Detail</NavLink>
        <NavLink to="/regime">Regime and Transitions</NavLink>
        <NavLink to="/verification">Verification</NavLink>
        <NavLink to="/model-info">Model Information</NavLink>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/districts" element={<DistrictDetail />} />
          <Route path="/districts/:districtId" element={<DistrictDetail />} />
          <Route path="/regime" element={<RegimeTransitions />} />
          <Route path="/verification" element={<Verification />} />
          <Route path="/model-info" element={<ModelInfo />} />
        </Routes>
      </main>
    </>
  )
}
