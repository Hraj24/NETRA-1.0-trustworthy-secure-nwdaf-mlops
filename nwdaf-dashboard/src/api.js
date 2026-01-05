import axios from "axios";

const API = "http://netra-1-0-trustworthy-secure-nwdaf-mlops.onrender.com/";

export const predict = (data) =>
  axios.post(`${API}/predict`, data);

export const explain = (data) =>
  axios.post(`${API}/explain`, data);

export const health = () =>
  axios.get(`${API}/health`);
