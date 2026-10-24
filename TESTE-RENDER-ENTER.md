# Teste: CPU ou GPU no Enter (renderizar efeitos e transições)

Vale para o Enter na linha do tempo (`Sequence ▸ Render Effects In to Out`), que renderiza os
trechos vermelhos e amarelos em verde.

## O que foi implementado

- Seletor no canto inferior direito da barra de status: `Enter renderiza: CPU` / `Enter renderiza: GPU`.
  Clique para alternar. A escolha é salva em `preferences.json`, em `timeline.renderBackend`
  (`cpu` | `gpu`). Padrão: `cpu`.
- `cpu`: comportamento de sempre (rota de referência, renderização por software).
- `gpu`: **falha fechada**. Nenhum quadro é renderizado e nada cai silenciosamente para a CPU.
  O motivo é que o pipeline de exportação ainda não tem uma rota GPU para prévias: o modo `Auto`
  é recusado e a rota residente AMD exige admissão que a prévia não pede.

## Como testar na sua máquina (AMD ou NVIDIA)

Pré-requisito: uma timeline com efeito ou transição que apareça vermelho/amarelo na barra.

1. Abra o projeto e confira o seletor no canto inferior direito. O padrão deve ser `Enter renderiza: CPU`.
2. Aperte Enter com o seletor em CPU. A barra deve ficar verde e o trecho renderizado deve ficar
   verde na régua.
3. Clique no seletor para `Enter renderiza: GPU`. Feche e reabra o FilmCraft: a escolha deve persistir.
4. Aperte Enter com GPU. Esperado hoje: uma mensagem na barra de status dizendo que a GPU não está
   disponível, **nenhum trecho fica verde** e nenhum quadro é renderizado pela CPU.
5. Volte o seletor para CPU antes de terminar a sessão.

Resultado esperado de GPU hoje: falha com mensagem. Se a barra ficar verde com GPU selecionada,
registre como defeito (violação de falha fechada).

## Sem medição

- Não há medição de paridade GPU x CPU: a rota GPU para prévias não existe ainda.
- Nenhum teste rodou em GPU real. Os testes automatizados cobrem a escolha, a validação, a
  persistência e a falha fechada, sem GPU.
