import { render } from '@testing-library/react';
import type { ReactElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import App from '../App';
import { AppProvider } from '../context/AppContext';

export function renderApp(path = '/') {
  return render(
    <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AppProvider>
        <App />
      </AppProvider>
    </MemoryRouter>,
  );
}

export function renderPage(ui: ReactElement, path = '/') {
  return render(
    <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AppProvider>
        <Routes><Route path="*" element={ui} /></Routes>
      </AppProvider>
    </MemoryRouter>,
  );
}
