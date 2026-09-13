function step2_inspect()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  try
    m=mphload([base 'ARC_TA_loss_Nc12_step1.mph']); fprintf('LOADED\n');
    Np=m.param.evaluate('Np'); Rin=m.param.evaluate('Rin'); R2=m.param.evaluate('R2');
    wid=m.param.evaluate('wid'); dist=m.param.evaluate('dist');
    fprintf('--- 几何参数 ---\n');
    fprintf('Np=%g (饼数), 半模型饼数 Np/2=%g\n', Np, Np/2);
    fprintf('Rin=%.3f R2=%.3f wid=%.4f dist=%.4f m\n', Rin,R2,wid,dist);
    Rlo=Rin; Rhi=Rin+R2; Zhi=(Np/2)*(wid+dist);
    fprintf('线圈范围: R=[%.2f, %.2f] m, z=[0, %.3f] m\n', Rlo, Rhi, Zhi);
    % 空气域 r2
    sz=m.geom('geom1').feature('r2').getDoubleArray('size'); ps=m.geom('geom1').feature('r2').getDoubleArray('pos');
    fprintf('空气域 r2: pos=[%.2f %.2f] size=[%.2f %.2f] -> R=[%.2f,%.2f] z=[%.2f,%.2f]\n', ...
            ps(1),ps(2),sz(1),sz(2), ps(1),ps(1)+sz(1), ps(2),ps(2)+sz(2));
    if Rhi > ps(1)+sz(1); fprintf('  !! 线圈 R_max=%.2f > 空气域 R_max=%.2f -> 超出, 需放大\n', Rhi, ps(1)+sz(1)); end
    % --- 选择集计数 (三段式验证) ---
    fprintf('--- 三段式选择集 (per-pancake = 数/%g) ---\n', Np/2);
    names={'uni3','uni5','uni4','uni8','uni7','uni2'}; labels={'n1内边缘','n2内过渡','n3中间','n1.1外边缘','n2.1外过渡','全部带材'};
    exp_pp=[30 7 20 30 7 94];
    for k=1:numel(names)
      try; s=mphgetselection(m.component('comp1').selection(names{k})); nb=numel(s.entities);
        fprintf('  %s(%s): %d 边界, per-pancake=%.1f (期望%d) %s\n', names{k}, labels{k}, nb, nb/(Np/2), exp_pp(k), ...
                tern(abs(nb/(Np/2)-exp_pp(k))<0.5,'OK','MISMATCH'));
      catch; fprintf('  %s: 计数失败\n', names{k}); end
    end
    % --- PMC / MI 边界 ---
    fprintf('--- mf 对称边界 ---\n');
    try; p=m.physics('mf').feature('pmc1').selection.inputentities; fprintf('  PMC pmc1 边界: %s\n', mat2str(double(p))); catch; fprintf('  pmc1 读取失败\n'); end
    try; q=m.physics('mf').feature('mi2').selection.inputentities; fprintf('  MI  mi2  边界: %s\n', mat2str(double(q))); catch; fprintf('  mi2 读取失败\n'); end
    fprintf('INSPECT_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
end
function s=tern(c,a,b); if c; s=a; else; s=b; end; end
