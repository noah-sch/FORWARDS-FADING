import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Home from './pages/Home';
import About from './pages/About';
import EliseLM from './pages/EliseLM';
import Pricing from './pages/Princing';
import Updates from './pages/Updates';
import Blog from './pages/Blog';

function App() {
    return (
        <>
            <div className='w-screen h-screen flex flex-col items-start justify-start bg-c4 scrollbar-hide'>
                <Router>
                    <Routes>
                        {/* Route Chat sans Layout (interface fullscreen) */}
                        <Route path="/eliselm" element={<EliseLM />} />

                        {/* Toutes les autres routes avec Layout */}
                        <Route path="/" element={<Layout><Home /></Layout>} />
                        <Route path="/about" element={<Layout><About /></Layout>} />
                        <Route path="/pricing" element={<Layout><Pricing /></Layout>} />
                        <Route path="/updates" element={<Layout><Updates /></Layout>} />
                        <Route path="/blog" element={<Layout><Blog /></Layout>} />
                    </Routes>
                </Router>
            </div>
        </>
    )
}

export default App
