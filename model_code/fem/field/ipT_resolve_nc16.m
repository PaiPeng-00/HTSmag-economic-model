function ipT_resolve_nc16()
  % arc_16_pancake 变体: Nc=16 (vs ARC 12). Heig=(wid+dist)*Nc 改变 -> 场改变 -> Ip(0.7) 改变。
  % 重建几何+网格+重解, 找各温度载流比0.7的 Ip, Nt=8.4MA/(16*Ip)。不覆盖原 ARC(Nc=12) mph。
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'ipT_nc16.log']);
  try
    m=mphload([base 'ARC_field_mangiarotti.mph']);
    Nc=16; AT=8.4e6; lf=0.7;
    m.param.set('Nc', num2str(Nc));     % 改饼数 -> Heig 变
    fprintf('Nc set to %d, rebuilding geom+mesh...\n', Nc);
    m.geom('geom1').run();              % 重建几何 (Heig 变)
    m.mesh('mesh1').run();              % 重建网格
    fprintf('=== Nc=16 Ip(T)/Nt(T) @载流比 %.2f (各温度重解刷新快照) ===\n', lf);
    fprintf(' T[K]  minJc[A/mm2]  Ic_tape[A]  Ip(0.7)[A]  Ip取整  Nt\n');
    M=[];
    for T=[4.2 10 20]
      m.param.set('Top', num2str(T));
      m.sol('sol1').runAll;
      Jcmin = mphmin(m,'Jc_mang','volume','selection',2,'dataset','dset1');
      Ic=1.2*Jcmin; Ip=lf*Ic; Ip_r=round(Ip/10)*10; Nt=AT/(Nc*Ip_r);
      fprintf(' %4.1f  %8.0f     %7.0f    %7.0f    %5d  %5.0f\n', T,Jcmin,Ic,Ip,Ip_r,Nt);
      M=[M; T Jcmin Ic Ip Ip_r Nt];
    end
    writematrix(M,[res 'ip_vs_T_nc16.csv']);
    fprintf('NC16_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
