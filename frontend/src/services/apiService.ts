const API_BASE_URL = "http://localhost:8000/api/v1";

const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'An API error occurred');
  }
  return response.json();
};

const apiService = {
  pauseTrading: async () => {
    const response = await fetch(`${API_BASE_URL}/actions/pause`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  resumeTrading: async () => {
    const response = await fetch(`${API_BASE_URL}/actions/resume`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  emergencyStop: async () => {
    const response = await fetch(`${API_BASE_URL}/actions/emergency_stop`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  closePosition: async (instrument: string) => {
    // The backend plan didn't include this endpoint yet, so we'll define it here
    // and implement it on the backend next.
    const response = await fetch(`${API_BASE_URL}/positions/${instrument}`, {
      method: 'DELETE',
    });
    return handleResponse(response);
  },
};

export default apiService;
