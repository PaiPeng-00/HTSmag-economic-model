function step3_ge_arc()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'step3_ge_arc.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_air.mph']); fprintf('LOADED\n');
    M=readmatrix([base 'ARC_M18.csv']);   % 18x18 ARC 互感
    neq=size(M,1);
    fprintf('ARC 矩阵 %dx%d, self范围[%.4e,%.4e]\n', neq,neq, min(diag(M)),max(diag(M)));
    % 替换前: 读当前(SPARC) self
    ge=m.component('comp1').physics('ge').feature('ge8');
    try; old=char(ge.getStringMatrix('equation')); catch; old=''; end
    % --- 重构 18 个方程 (每个方程用第 i 行 ARC 值) ---
    eq=cell(neq,1);
    for i=1:neq
      t='';
      for j=1:neq
        t=[t sprintf('%.6e[H]*Ia%dt', M(i,j), j)];
        if j<neq; t=[t ' + ']; end
      end
      eq{i}=sprintf('current(t) - (Nt/Npw*Np)^2/(Rcoil*Np)*(%s) - Ia%d', t, i);
    end
    ge.set('equation', eq);
    fprintf('ge8 已替换为 ARC 18 行\n');
    % --- 验证: 读回 eq1 ---
    back=ge.getStringMatrix('equation');
    e1=char(back(1));
    tok=regexp(e1,'([\d.]+e-0\d)\[H\]\*Ia1t','tokens','once');
    fprintf('验证 eq1 第1项(self): %s[H]  (应~1.479e-05 ARC, 非 5.818e-06 SPARC)\n', tok{1});
    e2=char(back(2)); tok2=regexp(e2,'([\d.]+e-0\d)\[H\]\*Ia2t','tokens','once');
    fprintf('验证 eq2 第2项(self): %s[H]  (应~1.479e-05)\n', tok2{1});
    mphsave(m,[base 'ARC_TA_loss_Nc12_geARC.mph']);
    fprintf('SAVED ARC_TA_loss_Nc12_geARC.mph\nGE_ARC_OK\n');
  catch ME
    fprintf(2,'GE_ARC_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
