// =============================
// CONFIGURAÇÕES DO SUPABASE
// =============================
const SUPABASE_URL = "https://fxafybrtuvxaxlrxhepq.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4YWZ5YnJ0dXZ4YXhscnhoZXBxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE4Njk3MDksImV4cCI6MjA3NzQ0NTcwOX0.F2V0DZYQV_2YkAQAzdwXlu_AbU1IFq49TE6PcihTk1c";

// Criação do cliente Supabase
const supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// =============================
// ELEMENTOS HTML
// =============================
const emailEl = document.getElementById("email");
const passEl = document.getElementById("password");
const msg = document.getElementById("msg");

// =============================
// LOGIN EXISTENTE
// =============================
document.getElementById("btnSignIn").addEventListener("click", async () => {
  msg.style.color = "red";
  msg.textContent = "";

  const email = emailEl.value.trim();
  const password = passEl.value;

  if (!email || !password) {
    msg.textContent = "Preencha email e senha.";
    return;
  }

  const { data, error } = await supabase.auth.signInWithPassword({ email, password });

  if (error) {
    msg.textContent = "Erro: " + error.message;
    return;
  }

  const userId = data.user.id;
  const accessToken = data.session?.access_token ?? "";

  msg.style.color = "green";
  msg.textContent = "Login realizado! Redirecionando...";

  // URL do seu Streamlit local
  const streamlitUrl = "http://localhost:8501/";

  // Redireciona para o chat passando o user_id e o token
  window.location.href =
    `${streamlitUrl}?user_id=${encodeURIComponent(userId)}&access_token=${encodeURIComponent(accessToken)}`;
});

// =============================
// CRIAR NOVA CONTA
// =============================
document.getElementById("btnSignUp").addEventListener("click", async () => {
  msg.style.color = "red";
  msg.textContent = "";

  const email = emailEl.value.trim();
  const password = passEl.value;

  if (!email || !password) {
    msg.textContent = "Preencha email e senha.";
    return;
  }

  const { data, error } = await supabase.auth.signUp({ email, password });

  if (error) {
    msg.textContent = "Erro: " + error.message;
    return;
  }

  msg.style.color = "green";
  msg.textContent = "Conta criada! Verifique seu email e faça login.";
});
