import axios from "axios";

const API = "http://127.0.0.1:8000";

export const predict = (data) =>
  axios.post(`${API}/predict`, data);

export const explain = (data) =>
  axios.post(`${API}/explain`, data);

export const health = () =>
  axios.get(`${API}/health`);
